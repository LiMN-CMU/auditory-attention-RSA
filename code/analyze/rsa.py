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
import pickle

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
    Sigma_x = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    var_s = float(np.var(s, ddof=1))
    inv_var_s = 1.0 / max(var_s, eps)

    # Haufe transform
    A = (Sigma_x @ W) * inv_var_s

    return A

def train_model_permutation(
        X, y, model_type, test_idx, 
        regularization_param=1, positive_label=1, 
        apply_direction_fix=True, apply_haufe_transform=True
    ):
    """
    Train a model on X, y with specified test_idx (leave-one-out style).
    Returns: weights, bias, decision values (logits), accuracy.
    
    Direction fix:
      Ensures mean decision value for `positive_label` on TRAIN set is larger than the other class.
      If not, flips (w, b) -> (-w, -b). This does NOT change predictions if you also flip decision values accordingly.
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
        w = model.coef_.ravel().copy()
        b = float(model.intercept_.ravel()[0])

        # decision values
        d_train = model.decision_function(X_train)
        d_test  = model.decision_function(X_test)

        acc = model.score(X_test, y_test)

    elif model_type == "logistic_regression":
        model = LogisticRegression(penalty='l2', C=1 / regularization_param, solver="liblinear")
        model.fit(X_train, y_train)
        w = model.coef_.ravel().copy()
        b = float(model.intercept_.ravel()[0])

        d_train = model.decision_function(X_train)  # raw logit
        d_test  = model.decision_function(X_test)

        y_pred = model.predict(X_test)
        acc = float(np.mean(y_pred == y_test))

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Direction fix 
    flipped = False
    if apply_direction_fix:
        classes = np.unique(y_train)
        if len(classes) != 2:
            raise ValueError("Direction fix assumes binary labels.")

        if positive_label not in classes:
            raise ValueError(f"positive_label={positive_label} not present in y_train classes={classes}")

        other_label = classes[0] if classes[1] == positive_label else classes[1]

        mu_pos = float(d_train[y_train == positive_label].mean())
        mu_other = float(d_train[y_train == other_label].mean())
        
        delta = mu_pos - mu_other

        # Want mu_pos > mu_other. If not, flip everything.
        if mu_pos < mu_other:
            flipped = True
            w *= -1.0
            b *= -1.0
            d_train *= -1.0
            d_test  *= -1.0
            acc = 1.0 - acc
    
    # Haufe pattern (use TRAIN data, scaled)
    haufe_w = None
    if apply_haufe_transform:
        haufe_w = haufe_transform(X_train, w)

    return w, haufe_w, b, d_test, acc, flipped

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
    feat_indices = config_rsa["frequency_band_indices"]
    model_type = config_rsa["decoder_model"]
    model_regularization_parameter = config_rsa["model_regularization_parameter"]
    apply_haufe_transform = bool(config_rsa["apply_haufe_transform"])
    apply_direction_fix = bool(config_rsa["apply_direction_fix"])

    sub_str = f"sub-{sub_i:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")

    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        print(f"=== Processing {epoch_type} epochs ===")
        features = []
        for feat_idx in feat_indices:
            # set the input folder
            if feat_idx == -1:  # epoch folder
                in_feat_dir = in_dir / "epoch"
            else:
                in_feat_dir = in_dir / "cwt"
                
            bids_in = BIDSPath(
                subject=sub_str.split('-')[1],
                task=task,
                processing=in_feat_dir.name,
                datatype="eeg",
                root=in_feat_dir,
                description=epoch_type
            )
            in_file = bids_in.fpath
            if feat_idx == -1:  # epoch folder
                print(f"\n=== Processing EEG amplitude ===")
                epochs = mne.read_epochs(in_file, preload=True)
                eeg_amp = epochs.get_data()  # (n_cond * n_trial, n_chan, n_time)
                
                label_fpath = in_file.parent / (in_file.stem.split("_desc")[0] + '_conditions.npy')
                condition_labels = np.load(label_fpath)
                conds = range(1, config_rsa["num_condition"] + 1)
                feat_erp = []
                for cond_i in conds:
                    condition_amp = eeg_amp[condition_labels == cond_i]  # Select trials for condition
                    condition_amp = condition_amp[:, :, :]
                    feat_erp.append(condition_amp)
                features.append(feat_erp)
            else:
                print(f"\n=== Processing frequency band #{feat_idx} ===")
                feat_full = np.load(in_file.with_suffix(".npy"), allow_pickle=True)  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)
                feat_band = []
                for cond_i in range(len(feat_full)):
                    feat_cond = feat_full[cond_i][:, :, feat_idx, :]
                    feat_band.append(feat_cond)
                features.append(feat_band)
        
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

        n_cond = config_rsa["num_condition"]
        if len(features) == 2:  # if multiple features:
            feat1, feat2 = features
            input_feature = []
            for c in range(n_cond):
                a = feat1[c]
                b = feat2[c]

                # basic checks: must match all dims except channel axis
                if a.ndim != b.ndim:
                    raise ValueError(f"cond {c}: ndim mismatch {a.ndim} vs {b.ndim}")

                input_feature.append(np.concatenate([a, b], axis=1))
        elif len(features) == 1:
            input_feature = features[0]
        else:
            raise Exception("The len(features) should be either 1 or 2")
        _, n_chan, n_time = input_feature[0].shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        
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
        all_weights_raw = np.zeros((n_time, n_cond, n_cond, n_chan))
        all_weights_haufe = np.zeros((n_time, n_cond, n_cond, n_chan))
        all_rdms = np.zeros((n_time, n_cond, n_cond))

        # Loop over time points
        for time_idx in time_indices:
            start_time = time.time()
            # cv_raw = {}  # key: (cond1, cond2) -> dict of arrays + metadata
            for cond1 in range(n_cond):
                for cond2 in range(cond1 + 1, n_cond):
                    trials1 = input_feature[cond1][:, :, time_idx]  # Shape: (n_trial, n_chan)
                    n_trial1 = trials1.shape[0]

                    trials2 = input_feature[cond2][:, :, time_idx]  # Shape: (n_trial, n_chan)
                    n_trial2 = trials2.shape[0]

                    X = np.vstack((trials1, trials2))
                    y = np.array([1] * n_trial1 + [0] * n_trial2)  # cond1: positive weight, cond2: negative weight
                    
                    # Generate reproducible test indices
                    test_indices = list(product(range(n_trial1), range(n_trial1, n_trial1 + n_trial2)))
                    # Parallel training (n_perm)
                    model_results = Parallel(n_jobs=-1)(
                        delayed(train_model_permutation)(
                            X, y, model_type,
                            test_idx=list(test_indices[perm_i]),
                            regularization_param=model_regularization_parameter,
                            apply_direction_fix=apply_direction_fix,
                            apply_haufe_transform=apply_haufe_transform,
                        )
                        for perm_i in range(len(test_indices))
                    )
                    # Extract weights and accuracies
                    weights_raw, weights_haufe, bias, test_logits, accuracies, sign_flipped = zip(*model_results)
                    weights_haufe = np.array(weights_haufe)  # Shape: (n_perm, n_chan)
                    accuracies = np.array(accuracies)  # Shape: (n_perm,)
                    weights_raw = np.array(weights_raw)  # Shape: (n_perm, n_chan)
                    bias = np.array(bias)  # Shape: (n_perm,)
                    test_logits = np.array(test_logits)  # Shape: (n_perm, n_test_labels)
                    sign_flipped = np.array(sign_flipped)
                    if sum(sign_flipped) > 0:
                        print(f"[WARNING] Sign flip detected: {sum(sign_flipped)}")

                    # stack cls data
                    all_weights_raw[time_idx, cond1, cond2, :] = weights_raw.mean(axis=0)
                    # symmetric
                    all_weights_raw[time_idx, cond2, cond1, :] = all_weights_raw[time_idx, cond1, cond2, :]
                    all_rdms[time_idx, cond1, cond2] = accuracies.mean()
                    all_rdms[time_idx, cond2, cond1] = all_rdms[time_idx, cond1, cond2]
                    
                    if apply_haufe_transform:
                        all_weights_haufe[time_idx, cond1, cond2, :] = weights_haufe.mean(axis=0)
                        all_weights_haufe[time_idx, cond2, cond1, :] = all_weights_haufe[time_idx, cond1, cond2, :]
            #         d = {
            #             "weights_raw":   np.asarray(weights_raw),     # (n_perm_local, n_chan)
            #             "weights_haufe": np.asarray(weights_haufe),   # (n_perm_local, n_chan)
            #             "bias":          np.asarray(bias),            # (n_perm_local,)
            #             "logits":        np.asarray(test_logits),     # (n_perm_local, 2)
            #             "acc":           np.asarray(accuracies),      # (n_perm_local,)
            #             "sign_flipped":  np.asarray(sign_flipped),    # (n_perm_local,)
            #             "n_trial1": n_trial1,
            #             "n_trial2": n_trial2,
            #             "time_idx": time_idx
            #         }
            #         cv_raw[(cond1, cond2)] = d

            # (out_file.parent / "model-cross-validation").mkdir(parents=True, exist_ok=True)
            # cv_path = (out_file.parent / "model-cross-validation" / f"{out_file.stem}_feat-{feat_idx}_model-{model_type}_config-{config_i}_alpha-{model_regularization_parameter}_time-{time_idx}_results.pkl")
            # with open(cv_path, "wb") as f:
            #     pickle.dump(cv_raw, f)
            process_time = time.time() - start_time
            print(f"Time taken: {process_time:.2f}s")

        # Save RDM & Weights
        feat_str = "-".join(map(str, feat_indices))
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_str}_model-{model_type}_config-{config_i}_alpha-{model_regularization_parameter}_rdm.npy", all_rdms)
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_str}_model-{model_type}_config-{config_i}_alpha-{model_regularization_parameter}_weights.npy", all_weights_raw)
        np.save(out_file.parent / f"{out_file.stem}_feat-{feat_str}_model-{model_type}_config-{config_i}_alpha-{model_regularization_parameter}_haufeweights.npy", all_weights_haufe)
    print(f"\nAll RDMs, weights, and accuracies saved for subject {sub_str}!\n")
    
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="analysis-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    print(config)
    for sub_i in config["subjects"]:
        run(config, sub_i)
