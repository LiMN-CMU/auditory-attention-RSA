import argparse
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath

# Parameters
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

def run(config, sub_i):
    task = config["task"]
    base_dir = Path(config["base_dir"])
    config_rsa = config["visualize_rsa"]
    in_dir = base_dir / config_rsa["input_folder"]
    out_dir = base_dir / config_rsa["output_folder"]
    
    fs = config_rsa["sampling_rate"]
    window_ms = config_rsa["target_time_window_ms"]  # ms window
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    
    feat_i = config_rsa["frequency_band_index"]
    breaks = config_rsa["rdm_plot_spacing_boundary"]  # where to insert spacing between groups
    
    montage_p = base_dir / "etc"
    with open(montage_p / "channel_dict_ABC.json", "r") as f:
        channel_mapping = json.load(f)
    montage = mne.channels.read_custom_montage(montage_p / "chanlocs_64_3_eye_chan.locs")
    ch_names = montage.ch_names[:64]  # Ensure matching length  TODO: n_channel hardcoded
    ch_types = ['eeg'] * 64
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    info.set_montage(montage)

    sub_str = f"sub-{sub_i:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")

    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
        acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]
        print(f"=== Processing {epoch_type} epochs ===")
        bids_in = BIDSPath(
            subject=sub_str.split('-')[1],
            task=task,
            processing=in_dir.name,
            datatype="eeg",
            root=in_dir,
            description=epoch_type
        )
        in_file = bids_in.fpath
        rdms = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_target-time-only_rdm.npy")
        svm_weights = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_target-time-only_svm-weights.npy")
        
        bids_out = BIDSPath(
            subject=sub_str.split('-')[1],
            task=task,
            processing=out_dir.name,
            datatype="eeg",
            root=out_dir,
            description=epoch_type
        )
        out_file = bids_out.fpath
        out_file.parent.mkdir(parents=True, exist_ok=True)

        n_time, n_cond, _ = rdms.shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        target_times = config_rsa[f"target_time_{epoch_type}"]
        target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in target_times]
        
        for i, target_time_idx in enumerate(target_time_indices):
            print(f"Processing {target_times[i]}")
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            
            # RDM
            window_rdms = rdms[window_start:window_end]
            target_rdm = np.mean(window_rdms, axis=0)

            # Add visual spacing between condition groups
            target_rdm_spaced = add_spacing(target_rdm, breaks)

            # Plot heatmap
            plt.figure(figsize=(8, 8))
            sns.heatmap(target_rdm_spaced, cmap="viridis", square=True,
                        vmin=acc_min, vmax=acc_max, xticklabels=False, yticklabels=False)
            plt.title(f"PCM-based RDM at time {target_times[i]} sec")
            plt.tight_layout()
            plt.savefig(out_file.parent / f"{sub_str}_task-{task}_desc-{epoch_type}_feat-{feat_i}_rdm-{target_times[i]:.1f}s.png", dpi=300)
            plt.close()
            
            # SVM Weights
            category_svm_weights = average_rdm_by_category(svm_weights)
            
            if svm_weights.shape[-1] == 64:  # TODO: hard-coded, why not interpolation?
                for cat_name, cat_weights in category_svm_weights.items():
                    window_svm_weights = cat_weights[window_start:window_end]
                    target_svm_weights = np.mean(window_svm_weights, axis=0)
                    
                    evoked = mne.EvokedArray(target_svm_weights[:, np.newaxis], info)  # Add time dimension
                    plt.figure(figsize=(8, 8))
                    evoked.plot_topomap(times=0, scalings=1, time_format='', cmap='RdBu_r', size=3, show=True)
                    fig_name = f"{sub_str}_task-{task}_desc-{epoch_type}_feat-{feat_i}_svm-weights_{target_times[i]:.1f}s_category-{cat_name}"
                    plt.title(fig_name)
                    plt.savefig(out_file.parent / f"{fig_name}.png", dpi=300)
                    plt.close()
                
            
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="analysis-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        run(config, sub_i)