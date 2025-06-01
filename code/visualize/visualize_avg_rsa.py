import argparse
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath

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

def average_rdm_by_category(weights, category_num_dict={"space": 8, "talker": 6, "relax": 7}):
    timepoints, n_cond, _, n_channels = weights.shape
    assert n_cond == sum(category_num_dict.values()), "Mismatch between RDM and labels"

    # Create list of condition indices per category
    cat_labels = np.concatenate([[i] * n for i, n in enumerate(category_num_dict.values())])
    cat_indices = [np.where(cat_labels == i)[0] for i in range(len(category_num_dict))]
    cat_names = list(category_num_dict.keys())

    # Helper to extract upper triangle RDM values for given condition index sets
    def extract_and_average(cond_idx1, cond_idx2, symmetric=True):
        mask = np.zeros((n_cond, n_cond), dtype=bool)
        for i in cond_idx1:
            for j in cond_idx2:
                if i != j and (not symmetric or i < j):
                    mask[i, j] = True
        return weights[:, mask, :]  # shape: (timepoints, num_pairs, n_channels)

    # Results stored in a dictionary
    results = {}

    # Within-category comparisons
    for i in range(len(cat_names)):
        key = f"{cat_names[i]}-{cat_names[i]}"
        results[key] = extract_and_average(cat_indices[i], cat_indices[i]).mean(axis=1)

    # Between-category comparisons
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            key = f"{cat_names[i]}-{cat_names[j]}"
            results[key] = extract_and_average(cat_indices[i], cat_indices[j], symmetric=False).mean(axis=1)

    return results  # Each entry has shape: (timepoints, n_channels)
  
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
        group_category_svm_weights = average_rdm_by_category(group_avg_svm_weights, category_num_dict=category_num_dict)
            
        for i, target_time_idx in enumerate(target_time_indices):
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            window_rdms = group_avg_rdms[window_start:window_end]
            target_rdm = np.mean(window_rdms, axis=0)

            # Add visual spacing between condition groups
            target_rdm_spaced = add_spacing(target_rdm, breaks)

            # Plot heatmap
            plt.figure(figsize=(8, 8))
            sns.heatmap(target_rdm_spaced, cmap="viridis", square=True,
                        vmin=acc_min, vmax=acc_max, xticklabels=False, yticklabels=False)
            # plt.title(f"PCM-based RDM at time {target_times[i]} sec")
            plt.tight_layout()
            out_file = out_dir / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_rdm_{target_times[i]:.1f}s.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            
            for cat_name, cat_weights in group_category_svm_weights.items():
                window_svm_weights = cat_weights[window_start:window_end]
                target_svm_weights = np.mean(window_svm_weights, axis=0)
                
                evoked = mne.EvokedArray(target_svm_weights[:, np.newaxis], info)  # Add time dimension
                plt.figure(figsize=(8, 8))
                weight_min = config_rsa[f"weight_boundary_{epoch_type}"][0]
                weight_max = config_rsa[f"weight_boundary_{epoch_type}"][1]
                evoked.plot_topomap(times=0, scalings=1, vlim=(weight_min, weight_max), time_format='', cmap='RdBu_r', size=3, show=True)
                fig_name = f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_weights_{target_times[i]:.1f}s_category-{cat_name}"
                # plt.title(fig_name)
                plt.savefig(out_file.parent / f"{fig_name}.png", dpi=300)
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