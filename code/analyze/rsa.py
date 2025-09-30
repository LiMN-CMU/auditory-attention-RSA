import argparse
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mne_bids import BIDSPath
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import time
from itertools import product
import mne
from datetime import datetime

# Function to train SVM and extract weights and accuracy
def train_model_permutation(X, y, model_type, test_idx, regularization_param=1):
    """
    Train a model (SVM or linear regression) on X, y with specified test_idx.
    Returns: weights and accuracy (for SVM) or R² (for regression).
    """
    X_train = np.delete(X, test_idx, axis=0)
    y_train = np.delete(y, test_idx)
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "svm":
        model = LinearSVC(C=regularization_param)
        model.fit(X_train, y_train)
        weights = model.coef_.flatten()
        logit = model.decision_function(X_test)[0]  # distance from decision threshold
        acc = model.score(X_test, y_test) 

    elif model_type == "logistic_regression":
        model = LogisticRegression(penalty='l2', C=1 / regularization_param, solver="liblinear")
        model.fit(X_train, y_train)
        weights = model.coef_.flatten()
        logit = model.decision_function(X_test)[0]  # raw logit value. sigmoid(logit) = probability
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test) 

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return weights, logit, acc

# Parameters
def run(config, sub_i):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_rsa = config["rsa"]
    in_dir = base_dir / config_rsa["input_folder"]
    out_dir = base_dir / config_rsa["output_folder"]
    config_i = config["configuration_id"]
    
    fs = config_rsa["sampling_rate"]
    window_ms = config_rsa["target_time_window_ms"]  # ms window
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    feat_idx = config_rsa["frequency_band_index"]
    model_type = config_rsa["decoder_model"]
    model_regularization_parameter = config_rsa["model_regularization_parameter"]

    sub_str = f"sub-{sub_i:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")

    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
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
        if config_rsa["input_folder"].startswith("derivatives/cwt"):
            power = np.load(in_file.with_suffix(".npy"))  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)
            print(f"\n=== Processing frequency band #{feat_idx} ===")
        elif config_rsa["input_folder"].startswith("derivatives/epoch"): 
            epochs = mne.read_epochs(in_file, preload=True)
            eeg_amp = epochs._data  # (n_cond * n_tial, n_chan, n_time)
            
            label_fpath = in_file.parent / (in_file.stem.split("_desc")[0] + '_conditions.npy')
            condition_labels = np.load(label_fpath)
            conds = range(1, config_rsa["num_condition"] + 1)
            power = []
            for cond_i in conds:
                condition_amp = eeg_amp[condition_labels == cond_i]  # Select trials for condition
                power.append(condition_amp)
            power = np.stack(power, axis=0)  
            power = power[:, :, :, np.newaxis, :]  # (n_cond, n_trial, n_chan, 1, n_time)
            print(f"\n=== Processing EEG amplitude ===")
        else:
            raise Exception("Input folder should be either cwt or epoch")
        
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
        
        # save config as json file within the output folder
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d%H%M%S")
        config_folder = out_file.parent / "config"
        config_folder.mkdir(parents=True, exist_ok=True)
        config_fpath = config_folder / f"config-{config_i}_time-{timestamp_str}.json"
        with open(config_fpath, "w") as f:
            json.dump(config, f)

        n_cond, n_trial, n_chan, n_feat, n_time = power.shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        
        # Generate reproducible test indices
        test_indices = list(product(range(n_trial), range(n_trial, 2 * n_trial)))
        
        target_times = config_rsa[f"target_time_{epoch_type}"]
        if isinstance(target_times, str) and target_times.lower() == "all":
            # Select every timepoint
            time_indices = np.arange(n_time)
        else:
            # Select specific target times
            target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in target_times]
            time_indices = []
            for target_time_idx in target_time_indices:
                window_start = max(0, target_time_idx - window_samples // 2)
                window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
                time_indices.append(np.arange(window_start, window_end))
            time_indices = np.concatenate(time_indices)

        # Storage for weights and accuracy across permutations
        all_weights = np.zeros((n_time, n_cond, n_cond, n_chan))
        all_logits = np.zeros((n_time, n_cond, n_cond))
        all_rdms = np.zeros((n_time, n_cond, n_cond))

        # Loop over features
        # for feat_idx in range(n_feat):
        for time_idx in time_indices:
            start_time = time.time()
            for cond1 in range(n_cond):
                for cond2 in range(cond1 + 1, n_cond):
                    trials1 = power[cond1, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
                    trials2 = power[cond2, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)

                    X = np.vstack((trials1, trials2))
                    y = np.array([1] * n_trial + [0] * n_trial)  # cond1: positive weight, cond2: negative weight
                    
                    # Parallel training (n_perm)
                    model_results = Parallel(n_jobs=-1)(
                        delayed(train_model_permutation)(
                            X, y, model_type, 
                            regularization_param=model_regularization_parameter, 
                            test_idx=list(test_indices[perm_i])) for perm_i in range(len(test_indices))
                    )

                    # Extract weights and accuracies
                    weights, logits, accuracies = zip(*model_results)
                    weights = np.array(weights)  # Shape: (n_perm, n_chan)
                    logits = np.array(logits)  # Shape: (n_perm, n_chan)
                    accuracies = np.array(accuracies)  # Shape: (n_perm,)

                    # Compute mean accuracy for RDM
                    # Store weights and accuracies
                    all_weights[time_idx, cond1, cond2, :] = np.mean(weights, axis=0)
                    all_logits[time_idx, cond1, cond2] = np.mean(logits, axis=0)
                    mean_acc = np.mean(accuracies)
                    all_rdms[time_idx, cond1, cond2] = mean_acc
                    all_rdms[time_idx, cond2, cond1] = mean_acc  # symmetric
                    
                    (out_file.parent / "model-cross-validation").mkdir(parents=True, exist_ok=True)
                    np.save(out_file.parent / "model-cross-validation" / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_time-{time_idx}_weights.npy", weights)
                    np.save(out_file.parent / "model-cross-validation" / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_time-{time_idx}_logits.npy", logits)
                    np.save(out_file.parent / "model-cross-validation" / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_time-{time_idx}_accuracies.npy", accuracies)

            process_time = time.time() - start_time
            print(f"Time taken: {process_time:.2f}s")

        # Save RDM
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_target-time-only_rdm.npy", all_rdms)

        # Save Weights & Accuracy
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_target-time-only_weights.npy", all_weights)
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_target-time-only_logits.npy", all_logits)

    print(f"\nAll RDMs, weights, and accuracies saved for subject {sub_str}!\n")
    
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
