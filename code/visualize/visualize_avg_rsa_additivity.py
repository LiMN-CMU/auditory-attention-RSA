import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from scipy import stats
from visualize_avg_rsa import add_spacing, average_by_category, run_temporal_cluster_permutation_tests, plot_dissimilarity_index, plot_significant_clusters, plot_rdms, category_dissimilarity_index_within_subject

# --- colorblind-safe palette ---
okabe_ito = [
    # "#E14949",
    # "#91C95A",
    # "#5BB5E5",
    "#F5A027",
    "#606563"
]

def run(config, sub_inds):
    task = config["task"]
    base_dir = Path(config["base_dir"])
    config_id = config["configuration_id"]
    config_rsa = config["visualize_rsa"]
    in_dir_multiple = base_dir / config_rsa["input_folder_multiple"]
    in_dir_single = base_dir / config_rsa["input_folder_single"]
    out_dir = base_dir / config_rsa["output_folder"]
    out_dir.mkdir(exist_ok=True, parents=True)
    
    model_type = config_rsa["decoder_model"]
    fs = config_rsa["sampling_rate"]
    window_ms = config_rsa["target_time_window_ms"]  # ms window
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    
    single_analysis_config_id = config_rsa["single_feature_analysis_config_id"]
    dual_analysis_config_id = config_rsa["dual_feature_analysis_config_id"]
    feat_indices = config_rsa["frequency_band_indices"]
    breaks = config_rsa["rdm_plot_spacing_boundary"]  # where to insert spacing between groups
    category_num_dict = config_rsa["category_number_dictionary"]
    attend_category_num_dict = config_rsa["attention_passive_category_number_dictionary"] if config_rsa.get("attention_passive_category_number_dictionary") else None
    categories_of_interest = config_rsa["categories_of_interest"]
    
    alpha = config_rsa["significance_alpha"]
    n_perm = config_rsa["significance_n_permutation"]
    n_corr = config_rsa["significance_n_bonferroni_correction"]
    cluster_p_threshold = config_rsa["significance_cluster_threshold_p_value"]
    
    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        print(f"=== Processing {epoch_type} epochs ===")
        acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
        acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]
        feat_1 = feat_indices[0]
        feat_2 = feat_indices[1]
        all_rdms = dict() # (n_subjects, n_time, n_cond, n_cond)
        
        group_rdm_name_1 = f"group_task-{task}_desc-{epoch_type}_feat-{feat_1}_model-{model_type}_config-{single_analysis_config_id}_rdm.npy"
        group_rdm_name_2 = f"group_task-{task}_desc-{epoch_type}_feat-{feat_2}_model-{model_type}_config-{single_analysis_config_id}_rdm.npy"
        group_rdm_name_mul = f"group_task-{task}_desc-{epoch_type}_feat-{feat_2}_model-{model_type}_config-{dual_analysis_config_id}_rdm.npy"
        if (in_dir_single / "group" / group_rdm_name_1).exists() and (in_dir_multiple / "group" / group_rdm_name_mul).exists():
            print(f"Loading stacked rdms & weights: {group_rdm_name_1}")
            all_rdms_1 = np.load(in_dir_single / "group" / group_rdm_name_1)
            all_rdms_2 = np.load(in_dir_single / "group" / group_rdm_name_2)
            all_rdms_mul = np.load(in_dir_multiple / "group" / group_rdm_name_mul)
            
            rdms_1_only = all_rdms_mul - all_rdms_2
            rdms_2_only = all_rdms_mul - all_rdms_1
            
            all_rdms[feat_1] = rdms_1_only
            all_rdms[feat_2] = rdms_2_only
        else:
            # load data
            for sub_i in sub_inds:
                sub_str = f"sub-{sub_i:03d}"
                print(f"\n=== Processing subject: {sub_str} ===")
                bids_in_single = BIDSPath(
                    subject=sub_str.split('-')[1],
                    task=task,
                    processing=in_dir_single.name,
                    datatype="eeg",
                    root=in_dir_single,
                    description=epoch_type
                )
                in_file_single = bids_in_single.fpath
                rdm_1 = np.load(in_file_single.parent / f"{in_file_single.stem}_feat-{feat_1}_model-{model_type}_config-{single_analysis_config_id}_target-time-only_rdm.npy")
                rdm_2 = np.load(in_file_single.parent / f"{in_file_single.stem}_feat-{feat_2}_model-{model_type}_config-{single_analysis_config_id}_target-time-only_rdm.npy")
                
                bids_in_multiple = BIDSPath(
                    subject=sub_str.split('-')[1],
                    task=task,
                    processing=in_dir_multiple.name,
                    datatype="eeg",
                    root=in_dir_multiple,
                    description=epoch_type
                )
                in_file_multiple = bids_in_multiple.fpath
                # feat_i in rsamultiple = index of the 2nd frequency band added to alpha 
                rdm_mul = np.load(in_file_multiple.parent / f"{in_file_multiple.stem}_feat-{feat_2}_model-{model_type}_config-{dual_analysis_config_id}_target-time-only_rdm.npy")
                
                rdm_2_only = rdm_mul - rdm_1
                rdm_1_only = rdm_mul - rdm_2
                
                all_rdms[feat_1].append(rdm_1_only)
                all_rdms[feat_2].append(rdm_2_only)

        feat_sim_indices = defaultdict(lambda: {"mean": [], "margin_error": []})
        for feat_idx in feat_indices:
            all_rdms_feat = np.array(all_rdms[feat_idx])  # shape: (n_sub, n_time, n_cond, n_cond)
            print(all_rdms_feat.shape)
            
            n_sub, n_time, n_cond, _ = all_rdms_feat.shape
            time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
            
            plot_target_times = config_rsa[f"plot_target_time_{epoch_type}"]
            plot_target_times_dict = {np.argmin(np.abs(time_vec - t)): t for t in plot_target_times}
                
            # average RDMs and weights by task category 
            all_category_rdms = average_by_category(all_rdms_feat, category_num_dict=category_num_dict)
            if attend_category_num_dict is not None:
                attend_category_rdms = average_by_category(all_rdms_feat, category_num_dict=attend_category_num_dict)
                all_category_rdms.update(attend_category_rdms.items())
            
            # average across subjects
            group_avg_rdms = np.mean(all_rdms_feat, axis=0)  # shape: (n_time, n_cond, n_cond)
            
            # select only the wanted pairs
            categories_of_interest_for_sim_indices = []
            for cat_pair in categories_of_interest:
                cat1, cat2 = cat_pair.split("-")
                categories_of_interest_for_sim_indices.append(cat_pair)
                categories_of_interest_for_sim_indices.append(f"{cat1}-{cat1}")  # add within pairs
                categories_of_interest_for_sim_indices.append(f"{cat2}-{cat2}")
            all_category_rdms = {k: v for k, v in all_category_rdms.items() if k in categories_of_interest_for_sim_indices}
            
            # gather subject-averaged dissimilarity index
            group_sim_indices_rdms = defaultdict(lambda: {"values": [], "mean": [], "margin_error": []})
            for time_idx in range(len(time_vec)):        
                # calculate dissimilarity index
                sim_indices_rdms = category_dissimilarity_index_within_subject(all_category_rdms, categories_of_interest, time_idx)
                for cat_names, cat_values in sim_indices_rdms.items():
                    group_sim_indices_rdms[cat_names]["values"].append(cat_values["values"])
                    group_sim_indices_rdms[cat_names]["mean"].append(cat_values["mean"])
                    group_sim_indices_rdms[cat_names]["margin_error"].append(cat_values["margin_error"])

                # plot
                if time_idx in plot_target_times_dict.keys():  # to plot & save
                    target_time = plot_target_times_dict[time_idx]
                    target_rdm = group_avg_rdms[time_idx]

                    # Add visual spacing between condition groups
                    target_rdm_spaced = add_spacing(target_rdm, breaks)

                    # 1. Plot RDMs
                    fig, ax = plot_rdms(target_rdm_spaced, acc_min, acc_max)
                    out_file = out_dir / f"group_task-{task}_desc-{epoch_type}_features-{feat_indices}_feat-{feat_idx}-only_model-{model_type}_config-{config_id}_rdm_{target_time:.1f}s.png"
                    out_file.parent.mkdir(exist_ok=True, parents=True)
                    fig.savefig(out_file, dpi=300, transparent=True)
                    plt.close(fig)
            # 3-2. Plot dissimilarity index (separately)
            df = len(sub_inds) - 1
            t_thresh = stats.t.ppf(1 - (cluster_p_threshold / 2), df)
            temporal_results = run_temporal_cluster_permutation_tests(
                group_sim_indices_rdms,
                n_permutations=n_perm,
                alpha=alpha,
                n_correction=n_corr,
                cluster_thres=t_thresh
            )
            fig, ax = plot_dissimilarity_index(
                group_sim_indices_rdms,
                n_times=len(time_vec),
                config_rsa=config_rsa,
                epoch_type=epoch_type, 
                figsize=(12, 2)
            )    
            start_time, end_time = config_rsa[f"epoch_boundary_{epoch_type}"]
            x_positions = np.linspace(start_time, end_time, len(time_vec))
            plot_significant_clusters(ax, x_positions, temporal_results, min_duration=0.03)  # 20ms, TODO: hard-coded
            
            out_file = out_dir / "group" / f"group_task-{task}_desc-{epoch_type}_features-{feat_indices}_feat-{feat_idx}_model-{model_type}_config-{config_id}_sim-index.png"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file, dpi=600, transparent=True)
            plt.close(fig)
        
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="visualize-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    run(config, config["subjects"])