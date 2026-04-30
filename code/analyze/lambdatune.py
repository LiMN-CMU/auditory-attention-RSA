import argparse
from pathlib import Path
from itertools import product, combinations
import json
from joblib import Parallel, delayed
import mne
from mne_bids import BIDSPath
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test) 
        # acc = model.score(X_test, y_test) 

    elif model_type == "logistic_regression":
        model = LogisticRegression(penalty='l2', C=1 / regularization_param, solver="liblinear")
        model.fit(X_train, y_train)
        weights = model.coef_.flatten()
        logit = model.decision_function(X_test)[0]  # raw logit value. sigmoid(logit) = probability
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test) 

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return y_pred, weights, logit, acc

def optimize_hyperparmeters(X, 
                            y,
                            model_type,
                            search_space:list, 
                            n_folds:int=10
                            ):
    # print(f"Optimizing hyperparameter among {search_space} using {n_folds}-fold cv")
    cv = KFold(n_splits=n_folds, shuffle=True) # inner cv for lambda optimization
    optimization_result = pd.DataFrame()

    for lambda_ in search_space:
        scores = []
        # print(f"Lambda: {lambda_:.2e}")
        for train_index, valid_index in cv.split(y):
            y_preds, weights, logit, acc = train_model_permutation(X, y, model_type, valid_index, regularization_param=lambda_)
            scores.append(acc) # average score across channels
        optimization_result[lambda_] = scores.mean()
    # print(optimization_result)
    return optimization_result

def run_one_permutation(perm_i, X, y, test_indices, 
                        model_type, search_space, n_folds_inner):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Liblinear failed to converge.*")

    test_idx = list(test_indices[perm_i])

    # Inner CV Data
    X_train_inner = np.delete(X, test_idx, axis=0)
    y_train_inner = np.delete(y, test_idx)

    # Hyperparameter tuning
    tuning_table = optimize_hyperparmeters(
        X_train_inner,
        y_train_inner,
        model_type=model_type,
        search_space=search_space,
        n_folds=n_folds_inner
    )

    return {
        "perm": perm_i,
        "tuning_table": tuning_table,
    }

# Parameters
def run(config, sub_i):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_lbd = config["lambda_tuning"]
    in_dir = base_dir / config_lbd["input_folder"]
    out_dir = base_dir / config_lbd["output_folder"]
    config_i = config["configuration_id"]
    
    n_fold = config_lbd["n_fold"]
    search_space = config_lbd["lambda_search_space"]
    feat_idx = config_lbd["frequency_band_index"]
    model_type = config_lbd["decoder_model"]
    fs = config_lbd["sampling_rate"]
    window_ms = config_lbd["target_time_window_ms"]  # ms window
    condition_indices = config_lbd["condition_subsamples"]
    window_samples = int((window_ms / 1000) * fs)  # Convert ms to samples
    
    sub_str = f"sub-{sub_i:03d}"
    epoch_type_dict = {'cue': config_lbd["epoch_boundary_cue"], 'target': config_lbd["epoch_boundary_target"]}
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
        
        if config_lbd["input_folder"].startswith("derivatives/cwt"):
            if sub_i == 3:
                power = np.load(in_file.with_suffix(".npy"), allow_pickle=True)  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)
            else:
                power = np.load(in_file.with_suffix(".npy"))  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)
            print(f"\n=== Processing frequency band #{feat_idx} ===")
        elif config_lbd["input_folder"].startswith("derivatives/epoch"): 
            epochs = mne.read_epochs(in_file, preload=True)
            eeg_amp = epochs._data  # (n_cond * n_tial, n_chan, n_time)
            
            label_fpath = in_file.parent / (in_file.stem.split("_desc")[0] + '_conditions.npy')
            condition_labels = np.load(label_fpath)
            conds = range(1, config_lbd["num_condition"] + 1)
            power = []
            if sub_i == 3:
                for cond_i in conds:
                    condition_amp = eeg_amp[condition_labels == cond_i]  # Select trials for condition
                    condition_amp = condition_amp[:, :, np.newaxis, :]  # (n_trial, n_chan, 1, n_time)
                    power.append(condition_amp)
            else:
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
        
        n_cond = len(power)
        _, n_chan, n_feat, n_time = power[0].shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        # Select specific target times
        target_times = config_lbd[f"target_time_{epoch_type}"]
        target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in target_times]
        time_indices = []
        for target_time_idx in target_time_indices:
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            time_indices.append(np.arange(window_start, window_end))
        time_indices = np.concatenate(time_indices)
        
        # n_trial_test = int(test_proportion * n_trial)
        # print(f"Number of test trials: {n_trial_test}")
        # test_idx = np.random.randint(0, n_trial, size=n_trial_test)    
        
        n_cond_sample = len(condition_indices)
        # test_indices = list(product(range(n_trial), range(n_trial, 2 * n_trial)))
        cond_pairs = list(combinations(range(n_cond_sample), 2))

        # Loop over features
        all_results = []
        all_scores_by_lambda = {_lambda: [] for _lambda in search_space}
        for _lambda in search_space:
            print(f"Lambda: {_lambda:.2e}")
            for time_i, time_idx in tqdm(enumerate(time_indices)):
                # start_time_idx = time_idx[0]
                # end_time_idx = time_idx[-1]
                for cond_pair_i, (cond1, cond2) in enumerate(cond_pairs):
                    if sub_i == 3:
                        trials1 = power[condition_indices[cond1]][:, :, feat_idx, time_idx]  # Shape: (n_trial_test, n_chan)
                        trials2 = power[condition_indices[cond2]][:, :, feat_idx, time_idx] 
                    else:
                        trials1 = power[condition_indices[cond1], :, :, feat_idx, time_idx]  # Shape: (n_trial_test, n_chan)
                        trials2 = power[condition_indices[cond2], :, :, feat_idx, time_idx] 
                    # treat values in different time points as separate trials
                    # trials1 = trials1.transpose(0, 2, 1)
                    # trials1 = trials1.reshape(-1, trials1.shape[-1])
                    # trials2 = trials2.transpose(0, 2, 1)
                    # trials2 = trials2.reshape(-1, trials2.shape[-1])
                    
                    # trials1 = trials1.mean(axis=2)
                    # trials2 = trials2.mean(axis=2)
                                    
                    n_trial1 = trials1.shape[0]
                    n_trial2 = trials2.shape[0]
                    X = np.vstack((trials1, trials2))
                    y = np.array([1] * n_trial1 + [0] * n_trial2)  # cond1: positive weight, cond2: negative weight
                    # print(f"X shape: {X.shape}")
                    # Generate reproducible test indices in each condition pair
                    test_indices = list(product(range(n_trial1), range(n_trial1, n_trial1 + n_trial2)))
                    
                    cv = KFold(n_splits=n_fold, shuffle=True) # inner cv for lambda optimization

                    scores = []
                    for train_index, valid_index in cv.split(y):
                        y_preds, weights, logit, acc = train_model_permutation(X, y, model_type, valid_index, regularization_param=_lambda)
                        scores.append(acc)
                    avg_score = sum(scores) / len(scores)

                    all_scores_by_lambda[_lambda].append(avg_score)  # add average cv score
        avg_score_by_lambda = dict()
        for _lambda, score_list in all_scores_by_lambda.items():
            avg_score = sum(score_list) / len(score_list)
            print(f"{_lambda}: {avg_score:.3f}")
            avg_score_by_lambda[_lambda] = avg_score
        return avg_score_by_lambda
                
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="analysis-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    lambda_dict_allsub = {_lambda: [] for _lambda in config["lambda_tuning"]["lambda_search_space"]}
    for sub_i in config["subjects"]:
        print(f"Processing subject: {sub_i}")
        avg_score_dict = run(config, sub_i)
        for _lambda, avg_score in avg_score_dict.items():
            lambda_dict_allsub[_lambda].append(avg_score)
            
    lambda_avg_dict = dict()
    for _lamb, avg_scores in lambda_dict_allsub.items():
        lambda_avg_dict[_lamb] = sum(avg_scores) / len(avg_scores)
        print(f"{_lamb}: {sum(avg_scores) / len(avg_scores)}")