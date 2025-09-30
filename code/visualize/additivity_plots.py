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
    
    feat_indices = config_rsa["frequency_band_indices"]
    breaks = config_rsa["rdm_plot_spacing_boundary"]  # where to insert spacing between groups
    
    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        print(f"=== Processing {epoch_type} epochs ===")
        acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
        acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]
        feat_1 = feat_indices[0]
        feat_2 = feat_indices[1]
        all_rdms = {feat_1: [], feat_2: []} # (n_subjects, n_time, n_cond, n_cond)
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
            rdm_1 = np.load(in_file_single.parent / f"{in_file_single.stem}_feat-{feat_1}_model-{model_type}_target-time-only_rdm.npy")
            rdm_2 = np.load(in_file_single.parent / f"{in_file_single.stem}_feat-{feat_2}_model-{model_type}_target-time-only_rdm.npy")
            
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
            rdm_mul = np.load(in_file_multiple.parent / f"{in_file_multiple.stem}_feat-{feat_2}_model-{model_type}_target-time-only_rdm.npy")
            
            rdm_2_only = rdm_mul - rdm_1
            rdm_1_only = rdm_mul - rdm_2
            
            all_rdms[feat_1].append(rdm_1_only)
            all_rdms[feat_2].append(rdm_2_only)

        for feat_idx in feat_indices:
            all_rdms_feat = np.array(all_rdms[feat_idx])  # shape: (n_sub, n_time, n_cond, n_cond)
            print(all_rdms_feat.shape)
            group_avg_rdms = np.mean(all_rdms_feat, axis=0)  # shape: (n_time, n_cond, n_cond)
            
            n_time, n_cond, _ = group_avg_rdms.shape
            time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
            target_times = config_rsa[f"target_time_{epoch_type}"]
            target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in target_times]
            
            for i, target_time_idx in enumerate(target_time_indices):
                window_start = max(0, target_time_idx - window_samples // 2)
                window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
                window_rdms = group_avg_rdms[window_start:window_end]
                target_rdm = np.mean(window_rdms, axis=0)
                
                mask = ~np.eye(target_rdm.shape[0], dtype=bool)
                rdm_avg_improvement = np.mean(target_rdm[mask]) * 100

                # Add visual spacing between condition groups
                target_rdm_spaced = add_spacing(target_rdm, breaks)

                # Plot heatmap
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(target_rdm_spaced, cmap="viridis", square=True, cbar=False,
                            vmin=acc_min, vmax=acc_max, xticklabels=False, yticklabels=False, ax=ax)
                ax.text(0.5, -0.04, f"{rdm_avg_improvement:.2f}%", transform=ax.transAxes,
                        ha="center", va="top", fontsize=40, clip_on=False)
                plt.tight_layout()
                # plt.title(f"PCM-based RDM at time {target_times[i]} sec")
                # plt.xlabel(f"{rdm_avg_improvement:.2f}%", fontsize=16, labelpad=20)
                out_file = out_dir / f"group_task-{task}_desc-{epoch_type}_features-{feat_indices}_feat-{feat_idx}-only_model-{model_type}_conf-{config_id}_rdm_{target_times[i]:.1f}s.png"
                plt.savefig(out_file, dpi=300)
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