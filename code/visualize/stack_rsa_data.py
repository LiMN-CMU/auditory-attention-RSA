import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import OAS
from joblib import Parallel, delayed

# def haufe_transform_permutation(X_chunk, W_chunk, feat_idx, eps=1e-12):
#     n_time, n_cond, n_cond, n_channel = W_chunk.shape
#     W_transformed = np.zeros((n_time, n_cond, n_cond, n_channel))
#     for time_idx in range(n_time):
#         for cond1 in range(n_cond):
#             for cond2 in range(cond1 + 1, n_cond):
#                 # trials1 = X_chunk[cond1, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
#                 trials1 = X_chunk[cond1][:, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
#                 # trials2 = X_chunk[cond2, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
#                 trials2 = X_chunk[cond2][:, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
#                 X = np.vstack((trials1, trials2))
                
#                 scaler = StandardScaler().fit(X)
#                 W_avg = W_chunk[time_idx, cond1, cond2, :]
#                 W_raw = W_avg / (scaler.scale_ + eps)  # effective raw-space decoder (approx)
#                 W_haufe = haufe_transform(X, W_raw, eps=eps)
                
#                 # X_scaled = StandardScaler().fit_transform(X)
#                 # W_haufe = haufe_transform(X_scaled, W_avg, eps=eps)
#                 W_transformed[time_idx, cond1, cond2, :] = W_haufe
#     return W_transformed

def haufe_transform(X, W, eps=1e-12):
    """
    Compute Haufe patterns from linear decoder weights.

    Parameters
    ----------
    X: (n_samples, n_features) in SAME space as w (e.g., scaled)
    w: (n_features,)
    ----------
    returns a: (n_features,)
    """

    # Center data
    Xc = X - X.mean(axis=0, keepdims=True)
    s = Xc @ W

    # Covariances
    # Sigma_x = OAS().fit(Xc).covariance_  # increase the stability of cov. matrix estimation
    Sigma_x = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    var_s = float(np.var(s, ddof=1))
    # var_s = float((s.T @ s) / (Xc.shape[0] - 1))
    inv_var_s = 1.0 / max(var_s, eps)

    # Haufe transform
    A = (Sigma_x @ W) * inv_var_s

    return A

def _haufe_one_job(X_chunk, W_chunk, feat_idx, time_idx, cond1, cond2, eps):
    trials1 = X_chunk[cond1][:, :, feat_idx, time_idx]  # (n_trial, n_chan)
    trials2 = X_chunk[cond2][:, :, feat_idx, time_idx]  # (n_trial, n_chan)
    X = np.vstack((trials1, trials2))                   # (n_samples, n_chan)

    scaler = StandardScaler().fit(X)
    W_avg = W_chunk[time_idx, cond1, cond2, :]
    # W_raw = W_avg / (scaler.scale_ + eps)
    X_scaled = scaler.transform(X)

    # A = haufe_transform(X, W_raw, eps=eps)
    A = haufe_transform(X_scaled, W_avg, eps=eps)
    return time_idx, cond1, cond2, A

def haufe_transform_permutation_parallel(
    X_chunk,
    W_chunk,
    feat_idx,
    eps=1e-12,
    n_jobs=-1,
    backend="loky",      # process-based; good for numpy/sklearn workloads
    verbose=5,
):
    n_time, n_cond, _, n_chan = W_chunk.shape
    W_transformed = np.zeros((n_time, n_cond, n_cond, n_chan), dtype=float)

    # Build job list (upper triangle only, like your original)
    jobs = [(t, c1, c2)
            for t in range(n_time)
            for c1 in range(n_cond)
            for c2 in range(c1 + 1, n_cond)]

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_haufe_one_job)(X_chunk, W_chunk, feat_idx, t, c1, c2, eps)
        for (t, c1, c2) in jobs
    )

    for t, c1, c2, A in results:
        W_transformed[t, c1, c2, :] = A

    return W_transformed

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
        all_svm_weights_haufe = []
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
            rdms = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_config-{analysis_config_id}_target-time-only_rdm.npy")
            svm_weights = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_config-{analysis_config_id}_target-time-only_weights.npy")
            # logits = np.load(in_file.parent / f"{in_file.stem}_feat-{feat_i}_model-{model_type}_config-{analysis_config_id}_target-time-only_logits.npy")
            
            bids_in_data = BIDSPath(
                subject=sub_str.split('-')[1],
                task=task,
                processing=orig_data_dir.name,
                datatype="eeg",
                root=orig_data_dir,
                description=epoch_type
            )
            in_data_file = bids_in_data.fpath
            if orig_data_dir.name.endswith("cwt"):
                power = np.load(in_data_file.with_suffix(".npy"), allow_pickle=True)  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)
            elif orig_data_dir.name.endswith("epoch"): 
                epochs = mne.read_epochs(in_data_file, preload=True)
                eeg_amp = epochs._data  # (n_cond * n_tial, n_chan, n_time)
                
                label_fpath = in_data_file.parent / (in_data_file.stem.split("_desc")[0] + '_conditions.npy')
                condition_labels = np.load(label_fpath)
                conds = range(1, config_rsa["num_condition"] + 1)
                power = []
                for cond_i in conds:
                    condition_amp = eeg_amp[condition_labels == cond_i]  # Select trials for condition
                    power.append(condition_amp[:, :, np.newaxis, :])
                # power = np.stack(power, axis=0)  
                # power = power[:, :, :, np.newaxis, :]  # (n_cond, n_trial, n_chan, 1, n_time)
            else:
                raise Exception("Input folder should be either cwt or epoch")
            
            # haufe transform to the svm weights
            # svm_weights_haufe = haufe_transform_permutation(power, svm_weights, feat_idx=feat_i, eps=1e-12)
            svm_weights_haufe = haufe_transform_permutation_parallel(
                power, svm_weights, feat_idx=feat_i, eps=1e-12,
                n_jobs=16
            )
            
            plt.hist(svm_weights[svm_weights != 0], bins=100, label="Raw", alpha=0.5)
            plt.hist(svm_weights_haufe[svm_weights_haufe != 0], bins=100, label="Haufe", alpha=0.5)
            # plt.xlim([-3, 3])
            plt.legend()
            plt.savefig(f'Haufe_comparisons_{sub_str}.png')
            plt.close()

            all_rdms.append(rdms)
            all_svm_weights.append(svm_weights)
            all_svm_weights_haufe.append(svm_weights_haufe)

        all_rdms = np.array(all_rdms)  # shape: (n_sub, n_time, n_cond, n_cond)
        all_svm_weights = np.array(all_svm_weights)  # shape: (n_sub, n_time, n_cond, n_cond, n_channel)
        if all_svm_weights.shape[-1] != 64:
            all_svm_weights = all_svm_weights.reshape(all_svm_weights.shape[:-1] + (2, 64))  # TODO: n_channel hard-coded
            all_svm_weights = all_svm_weights.mean(axis=-2)
            
        np.save(in_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm", all_rdms)
        np.save(in_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_weights", all_svm_weights)
        np.save(in_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights", all_svm_weights_haufe)
        

if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="visualize-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id
    
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    print(config)
    run(config, config["subjects"])