import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from mne.channels import make_standard_montage, find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency

# --- Okabe–Ito colorblind-safe palette ---
okabe_ito = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7"   # reddish purple
]

def add_spacing(matrix, break_indices):
    size = matrix.shape[0]
    new_size = size + len(break_indices)
    new_matrix = np.full((new_size, new_size), np.nan)
    row_idx, col_idx = 0, 0
    breaks_set = set(break_indices)

    for i in range(new_size):
        if i in breaks_set:
            continue
        for j in range(new_size):
            if j in breaks_set:
                continue
            new_matrix[i, j] = matrix[row_idx, col_idx]
            col_idx += 1
        row_idx += 1
        col_idx = 0
    return new_matrix

def average_by_category(data, category_num_dict={"space": 8, "talker": 6, "relax": 7}):
    """
    Average between/within category values for time-resolved RDMs or weight matrices.

    Parameters
    ----------
    data : np.ndarray
        Shape:
            (timepoints, n_cond, n_cond)                  # RDMs
            (timepoints, n_cond, n_cond, n_channels)      # weights
        Must be symmetric with zeros on the diagonal (RDM case).

    category_num_dict : dict
        Keys: category names
        Values: number of conditions in each category.

    Returns
    -------
    results : dict
        Keys: 'catA-catB'
        Values:
            Shape (timepoints, n_channels) if weights
            Shape (timepoints,) if RDMs
    """
    ndim = data.ndim
    if ndim == 3:
        timepoints, n_cond, _ = data.shape
        n_channels = None
    elif ndim == 4:
        timepoints, n_cond, _, n_channels = data.shape
    else:
        raise ValueError(f"Data must have shape (time, cond, cond) or (time, cond, cond, channels), got {data.shape}")

    assert n_cond == sum(category_num_dict.values()), \
        f"Mismatch: {n_cond} conditions vs {sum(category_num_dict.values())} from category counts"

    # Condition indices for each category
    cat_labels = np.concatenate([[i] * n for i, n in enumerate(category_num_dict.values())])
    cat_indices = [np.where(cat_labels == i)[0] for i in range(len(category_num_dict))]
    cat_names = list(category_num_dict.keys())

    def extract_and_average(cond_idx1, cond_idx2, symmetric=True):
        mask = np.zeros((n_cond, n_cond), dtype=bool)
        for i in cond_idx1:
            for j in cond_idx2:
                if i != j and (not symmetric or i < j):
                    mask[i, j] = True
        if n_channels is None:
            return data[:, mask]  # (time, num_pairs)
        else:
            return data[:, mask, :]  # (time, num_pairs, channels)

    results = {}

    # Within-category
    for i in range(len(cat_names)):
        key = f"{cat_names[i]}-{cat_names[i]}"
        vals = extract_and_average(cat_indices[i], cat_indices[i])
        results[key] = vals.mean(axis=1)

    # Between-category
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            key = f"{cat_names[i]}-{cat_names[j]}"
            vals = extract_and_average(cat_indices[i], cat_indices[j], symmetric=False)
            results[key] = vals.mean(axis=1)

    return results

def category_dissimilarity_index(avg_results, category_names, time_indices):
    start_i, end_i = time_indices
    sim_results = {}

    for i in range(len(category_names)):
        for j in range(i + 1, len(category_names)):
            between_key = f"{category_names[i]}-{category_names[j]}"
            within_A = avg_results[f"{category_names[i]}-{category_names[i]}"][start_i: end_i]
            within_B = avg_results[f"{category_names[j]}-{category_names[j]}"][start_i: end_i]
            between_AB = avg_results[between_key][start_i: end_i]
            sim_index_vals = between_AB - 0.5 * (within_A + within_B)
            sim_results[between_key] = sim_index_vals.mean()

    return sim_results

def window_cluster_mask(results, category_name, time_idx, alpha=0.05, which="within"):
    """
    Build a boolean mask over channels indicating membership in any
    significant spatiotemporal cluster that overlaps [t_start:t_end).

    results: dict produced earlier (with 'per_cond' entries)
    category_name
    alpha: significance threshold
    Returns: mask_ch (n_chan,) boolean
    """
    rec = results["per_cond"][category_name]
    pvals = rec["cluster_p_values"]

    sig_idxs = [k for k, p in enumerate(pvals) if p <= alpha]
    if not sig_idxs:
        # print(f"No significant clusters for {category_name} at time {time_idx}")
        return None

    # Union across all significant clusters that touch any time in the window
    n_time, n_chan = rec["clusters_2d"][0].shape
    mask_ch = np.zeros(n_chan, dtype=bool)
    for k in sig_idxs:
        cl = rec["clusters_2d"][k]  # shape (n_time, n_chan), bool
        mask_ch |= cl[time_idx, :]  # directly take this row

    if not mask_ch.any():
        # print(f"No cluster touches time {time_idx} for {category_name}")
        return None
    
    print(f"Cluster touches time {time_idx} for {category_name}")
    return mask_ch

def run(config, sub_inds):
    task = config["task"]
    base_dir = Path(config["base_dir"])
    config_id = config["configuration_id"]
    config_rsa = config["visualize_rsa"]
    in_dir = base_dir / config_rsa["input_folder"]
    out_dir = base_dir / config_rsa["output_folder"]
    
    model_type = config_rsa["decoder_model"]
    fs = config_rsa["sampling_rate"]
    window_ms = config_rsa["target_time_window_ms"]  # ms window
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    
    feat_i = config_rsa["frequency_band_index"]
    breaks = config_rsa["rdm_plot_spacing_boundary"]  # where to insert spacing between groups
    category_num_dict = config_rsa["category_number_dictionary"]
    
    montage_p = base_dir / "etc"
    with open(montage_p / "channel_dict_ABC.json", "r") as f:
        channel_mapping = json.load(f)
    montage = mne.channels.read_custom_montage(montage_p / "chanlocs_64_3_eye_chan.locs")
    ch_names = montage.ch_names[:64]  # Ensure matching length  TODO: n_channel hardcoded
    ch_types = ['eeg'] * 64
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    info.set_montage(montage)

    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        print(f"=== Processing {epoch_type} epochs ===")
        acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
        acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]
        all_rdms = [] # (n_subjects, n_time, n_cond, n_cond)
        all_svm_weights = []
        for sub_i in sub_inds:
            sub_str = f"sub-{sub_i:03d}"
            print(f"\n=== Processing subject: {sub_str} ===")
            bids_in = BIDSPath(
                subject=sub_str.split('-')[1],
                task=task,
                processing=in_dir.name,
                datatype="eeg",
                root=in_dir,
                description=epoch_type
            )
            in_file = bids_in.fpath
            rdms = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_target-time-only_rdm.npy")
            # rdms = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_target-time-only_rdm.npy")
            svm_weights = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_target-time-only_weights.npy")
            # svm_weights = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_target-time-only_svm-weights.npy")
            if svm_weights.shape[-1] != 64:  # TODO: hard-coded, why not interpolation?
                continue
            print(rdms.shape)
            print(svm_weights.shape)
            all_rdms.append(rdms)
            all_svm_weights.append(svm_weights)

        all_rdms = np.array(all_rdms)  # shape: (n_sub, n_time, n_cond, n_cond)
        all_svm_weights = np.array(all_svm_weights)  # shape: (n_sub, n_time, n_cond, n_cond, n_channel)
        print(all_rdms.shape)
        print(all_svm_weights.shape)
        group_avg_rdms = np.mean(all_rdms, axis=0)  # shape: (n_time, n_cond, n_cond)
        group_avg_svm_weights = np.mean(all_svm_weights, axis=0)
        
        n_time, n_cond, _ = group_avg_rdms.shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        target_times = config_rsa[f"target_time_{epoch_type}"]
        target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in target_times]
        
        # SVM Weights
        group_category_svm_weights = average_by_category(group_avg_svm_weights, category_num_dict=category_num_dict)
        group_category_rdms = average_by_category(group_avg_rdms, category_num_dict=category_num_dict)
        group_sim_indices_rdms = defaultdict(list)

        # weight matrix for permutation testing (keep the subject axis intact)
        category_svm_weights_full = defaultdict(list)
        for sub_i in range(len(sub_inds)):
            all_svm_weight_sub = all_svm_weights[sub_i]
            # category averaging within a subject
            category_svm_weights_sub = average_by_category(all_svm_weight_sub, category_num_dict=category_num_dict)
            for cat_name, cat_val in category_svm_weights_sub.items():
                # add the value
                category_svm_weights_full[cat_name].append(cat_val)
        for cat_name, cat_val in category_svm_weights_full.items():
            category_svm_weights_full[cat_name] = np.stack(category_svm_weights_full[cat_name], axis=0)  # 'n_category_pair': (n_sub, n_time, n_channel)
            
        category_svm_weights_target = defaultdict(list)
        for time_i, target_time_idx in enumerate(target_time_indices):
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            window_rdms = group_avg_rdms[window_start:window_end]
            target_rdm = np.mean(window_rdms, axis=0)

            # Add visual spacing between condition groups
            target_rdm_spaced = add_spacing(target_rdm, breaks)

            # Plot heatmap
            plt.figure(figsize=(8, 8))
            sns.heatmap(target_rdm_spaced, cmap="viridis", square=True, cbar=False,
                        vmin=acc_min, vmax=acc_max, xticklabels=False, yticklabels=False)
            # plt.title(f"PCM-based RDM at time {target_times[time_i]} sec")
            plt.tight_layout()
            out_file = out_dir / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_rdm_{target_times[time_i]:.1f}s.png"
            plt.savefig(out_file, dpi=300, transparent=True)
            plt.close()
            
            for cat_name, cat_weights in group_category_svm_weights.items():
                window_svm_weights = cat_weights[window_start:window_end]
                target_svm_weights = np.mean(window_svm_weights, axis=0)
                
                evoked = mne.EvokedArray(target_svm_weights[:, np.newaxis], info)  # Add time dimension
                plt.figure(figsize=(8, 8))
                weight_min = config_rsa[f"weight_boundary_{epoch_type}"][0]
                weight_max = config_rsa[f"weight_boundary_{epoch_type}"][1]
                evoked.plot_topomap(
                    times=0, 
                    scalings=1, 
                    vlim=(weight_min, weight_max),
                    time_format='',
                    size=3,
                    colorbar=False
                )
                fig_name = f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_weights_{target_times[time_i]:.1f}s_category-{cat_name}"
                # plt.title(fig_name)
                plt.savefig(out_file.parent / f"{fig_name}.png", dpi=300, transparent=True)
                plt.close()
            
            # dissimilarity index
            sim_indices_rdms = category_dissimilarity_index(group_category_rdms, list(category_num_dict.keys()), [window_start, window_end])
            for cat_names, cat_value in sim_indices_rdms.items():
                group_sim_indices_rdms[cat_names].append(cat_value)
            
            # cluster-based permutation analysis
            for cat_name, cat_val in category_svm_weights_full.items():
                target_val = cat_val[:, window_start:window_end, :]
                target_avg = np.mean(target_val, axis=1)
                category_svm_weights_target[cat_name].append(target_avg) 
        for cat_name, cat_val in category_svm_weights_target.items():
            category_svm_weights_target[cat_name] = np.stack(category_svm_weights_target[cat_name], axis=1)  # 'n_category_pair': (n_sub, n_target_time, n_channel)
        
        # cluster based permutation test
        alpha = 0.05 / 6 # TODO: hard-coded bonferrnoi correction
        n_permutations = 1000   # tune for speed/precision
        rng = np.random.RandomState(42)

        ch_adj, _ = find_ch_adjacency(info, ch_type="eeg")
        
        # Run one test per condition
        results = {
            "per_cond": dict(),         # one dict per condition
            "all_cluster_ps": list(),   # for optional across-condition correction
        }
        
        for cat_name, cat_val in category_svm_weights_target.items():
            n_sub, n_target_time, n_chan = cat_val.shape
            # spatiotemporal TFCE cluster one-sample test vs 0 across subjects
            T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
                cat_val,
                threshold=None,            # None => TFCE
                tail=0,
                n_permutations=n_permutations,
                adjacency=ch_adj,          # allows clusters over time × channels
                out_type="mask",
                n_jobs=1,
                seed=rng
            )
            # Store
            rec = {
                "T_obs": T_obs.reshape(n_target_time, n_chan),                # TFCE-enhanced stat map
                "clusters": clusters,                                  # list of boolean masks over (n_time*n_chan,)
                "clusters_2d": [c.reshape(n_target_time, n_chan) for c in clusters],
                "cluster_p_values": cluster_p                          # FWER-corrected within this condition
            }
            results["per_cond"][cat_name] = rec

            for k, p in enumerate(cluster_p):
                results["all_cluster_ps"].append(p)
        results["all_cluster_ps"] = np.array(results["all_cluster_ps"])
        breakpoint()
        
        # Attach corrected p-values back to each condition’s clusters
        # TODO: write
        
        # plot masked svm weights map
        for time_i, target_time_idx in enumerate(target_time_indices):
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            
            for cat_name, cat_weights in group_category_svm_weights.items():
                window_svm_weights = cat_weights[window_start:window_end]
                target_svm_weights = np.mean(window_svm_weights, axis=0)
                
                # Build significant-channel mask for this window from TFCE results:
                mask_ch = window_cluster_mask(results, cat_name, time_i, alpha=alpha)

                plt.figure(figsize=(8, 8))
                weight_min = config_rsa[f"weight_boundary_{epoch_type}"][0]
                weight_max = config_rsa[f"weight_boundary_{epoch_type}"][1]
                
                mask = None if mask_ch is None else mask_ch[:, np.newaxis]
                mask_params = dict(marker='o',
                                markerfacecolor='none',
                                markeredgecolor='k',
                                markersize=8)

                evoked = mne.EvokedArray(target_svm_weights[:, np.newaxis], info)  # Add time dimension
                evoked.plot_topomap(
                    times=0, 
                    scalings=1, 
                    vlim=(weight_min, weight_max),
                    time_format='',
                    size=3,
                    colorbar=False,
                    mask=mask,
                    mask_params=mask_params
                )

                fig_name = f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_weights_{target_times[time_i]:.1f}s_category-{cat_name}-masked"
                plt.savefig(out_file.parent / f"{fig_name}.png", dpi=300, transparent=False)
                plt.close()
            
        # plt.figure(figsize=(8, 5))
        # for cat_names, values in group_sim_indices_rdms.items():
        #     plt.plot(target_times, values, marker='o', label=cat_names)

        # plt.xlabel("Time")
        # plt.xticks(target_times * 1000)  # ms
        # plt.ylabel("Similarity Index")
        # plt.title("Category Similarity Indices over Time")
        # plt.legend()
        # plt.tight_layout()
        # out_file = out_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_sim-index.png"
        # plt.savefig(out_file, dpi=300)
        # plt.close()
        
        plt.figure(figsize=(12, 2))

        # Evenly spaced positions for the x-axis (but keep original labels)
        x_positions = np.linspace(0, len(target_times) - 1, len(target_times))

        for i, (cat_names, values) in enumerate(group_sim_indices_rdms.items()):
            color = okabe_ito[i % len(okabe_ito)]
            plt.plot(x_positions, values, marker='o', label=cat_names, color=color)

        # plt.xlabel("Time (ms)")
        # plt.xticks(x_positions, labels=[t * 1000 for t in target_times])
        # plt.ylabel("Similarity Index")
        ax = plt.gca()
        ax.set_xticks([])          # remove tick positions
        ax.set_xticklabels([])     # remove tick labels
        # ax.spines['bottom'].set_visible(False)  # hide x-axis line
        ax.set_xlabel("")          # remove x-axis label
        
        sim_y_min_max = config_rsa[f"similarity_index_boundary_{epoch_type}"]
        plt.ylim(sim_y_min_max[0], sim_y_min_max[1])  # Example: 0–1 range

        # Apply new legend order
        # ax = plt.gca()
        # handles, labels = ax.get_legend_handles_labels()
        # order = [1, 2, 0]
        # ax.legend([handles[idx] for idx in order],
        #         [labels[idx] for idx in order],
        #         loc="upper left")

        plt.tight_layout()
        out_file = out_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_sim-index.png"
        plt.savefig(out_file, dpi=600, transparent=True)
        plt.close()
            

if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="visualize-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    run(config, config["subjects"])