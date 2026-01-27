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
from mne.stats import spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test
from scipy import stats

def run(config, sub_inds):
    task = config["task"]
    base_dir = Path(config["base_dir"])
    config_id = config["configuration_id"]
    config_rsa = config["visualize_rsa"]
    in_dir = base_dir / config_rsa["input_folder"]
    orig_data_dir = base_dir / config_rsa["original_data_folder"]
    out_dir = base_dir / config_rsa["output_folder"]
    
    analysis_config_id = config_rsa["analysis-config-id"]
    model_type = config_rsa["decoder_model"]
    fs = config_rsa["sampling_rate"]
    window_ms = config_rsa["target_time_window_ms"]  # ms window
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    
    feat_i = config_rsa["frequency_band_index"]
    breaks = config_rsa["rdm_plot_spacing_boundary"]  # where to insert spacing between groups
    category_num_dict = config_rsa["category_number_dictionary"]
    
    lambda_range = config_rsa["model_regularization_parameter_log_range"]
    
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
    
    n_lamb = lambda_range[1] - lambda_range[0] + 1
    lambdas = np.logspace(lambda_range[0], lambda_range[1], num=n_lamb)
    rdm_dict = {'cue': {}, 'target': {}}
    for _lamb in lambdas:
        print(f"=== Processing lambda: {_lamb} ===")
        for epoch_type, epoch_boundary in epoch_type_dict.items():
            print(f"=== Processing {epoch_type} epochs ===")
            acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
            acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]
            all_rdms = [] # (n_subjects, n_time, n_cond, n_cond)
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
                rdms = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_config-{analysis_config_id}_alpha-{_lamb}_target-time-only_rdm.npy")
                
                all_rdms.append(rdms)

            all_rdms = np.array(all_rdms)  # shape: (n_sub, n_time, n_cond, n_cond)
            print(all_rdms.shape)
            
            n_sub, n_time, n_cond, _ = all_rdms.shape
            time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
            
            target_time = config_rsa[f"target_time_{epoch_type}"]
            target_time_idx = np.argmin(np.abs(time_vec - target_time))
            
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            
            # group_avg_rdms = np.mean(all_rdms, axis=0)  # shape: (n_time, n_cond, n_cond)
            window_rdms = all_rdms[:, window_start:window_end, :, :]  # shape: (n_sub, n_time, n_cond, n_cond)
            target_rdm = np.mean(window_rdms, axis=1)
            
            # select only the upper triangle values
            n_sub, n_cond, _ = target_rdm.shape
            row_indices, col_indices = np.triu_indices(n_cond, k=1)
            unique_target_rdm = target_rdm[:, row_indices, col_indices]
            avg_target_rdm = np.mean(unique_target_rdm, axis=0)  # average subjects
            
            rdm_dict[epoch_type][_lamb] = avg_target_rdm
            
            
    cue_rdms = list(rdm_dict['cue'].values())
    target_rdms = list(rdm_dict['target'].values())
    cue_corrs = np.corrcoef(cue_rdms)
    target_corrs = np.corrcoef(target_rdms)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cue_corrs, annot=True, vmin=-1, vmax=1, xticklabels=lambdas, yticklabels=lambdas, cmap='RdBu_r')
    plt.ylabel("lambda")
    plt.xlabel("lambda")
    plt.title("[Cue] Correlation between RDMs across lambdas")
    plt.savefig(out_dir / "group" / f'feat-{feat_i}_epoch-cue_lambda-correlations.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_corrs, annot=True, vmin=-1, vmax=1, xticklabels=lambdas, yticklabels=lambdas, cmap='RdBu_r')
    plt.ylabel("lambda")
    plt.xlabel("lambda")
    plt.title("[Stimulus] Correlation between RDMs across lambdas")
    plt.savefig(out_dir / "group" / f'feat-{feat_i}_epoch-target_lambda-correlations.png')
    plt.close()        
    
    row_indices, col_indices = np.triu_indices(cue_corrs.shape[0], k=1)
    
    unique_cue_corrs = cue_corrs[row_indices, col_indices]
    unique_target_corrs = target_corrs[row_indices, col_indices]
    
    print("=== Cue Period ===")
    print(f"Mean: {np.mean(unique_cue_corrs):.3f}")
    print(f"Median: {np.median(unique_cue_corrs):.3f}")
    print(f"Standard Deviation: {np.std(unique_cue_corrs):.3f}")
    print("=== Stimlus Period ===")
    print(f"Mean: {np.mean(unique_target_corrs):.3f}")
    print(f"Median: {np.median(unique_target_corrs):.3f}")
    print(f"Standard Deviation: {np.std(unique_target_corrs):.3f}")
        
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