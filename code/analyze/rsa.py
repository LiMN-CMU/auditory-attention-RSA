from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import time

# Parameters
subjects = [2]
task = "craa"
base_dir = Path("..") / "data" / "derivatives"
in_folder = "cwt"

n_permutations = 1000  # Number of leave-one-trial-out iterations

# Function to train SVM and extract weights and accuracy
def train_svm_permutation(X, y):
    """
    Train an SVM on data X with labels y using a two-trial test split (one from each class).
    Returns: SVM weights and accuracy.
    """
    # Pick one trial from each class
    test_idx_0 = np.random.randint(0, len(y) // 2)  # From class 0
    test_idx_1 = np.random.randint(len(y) // 2, len(y))  # From class 1
    test_indices = [test_idx_0, test_idx_1]

    X_train = np.delete(X, test_indices, axis=0)
    y_train = np.delete(y, test_indices)
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Train SVM
    svm = LinearSVC()
    svm.fit(X_train, y_train)

    # Get weight coefficients
    weights = svm.coef_.flatten()

    # Compute accuracy
    acc = svm.score(X_test, y_test)

    return weights, acc


for sub_id in subjects:
    sub_str = f"sub-{sub_id:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")

    in_p = base_dir / in_folder / sub_str / "eeg"
    out_p = in_p / "rdm_results"
    out_p.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    in_file_cue = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-target.npy"
    power_cue = np.load(in_file_cue)  # Shape: (n_cond, n_trial, n_chan, n_feat, n_time)

    n_cond, n_trial, n_chan, n_feat, n_time = power_cue.shape
    time_vec = np.linspace(-1.3, 1, n_time)

    # Storage for SVM weights and accuracy across permutations
    all_svm_weights = np.zeros((n_feat, n_chan, n_permutations))
    all_svm_accuracies = np.zeros((n_feat, n_permutations))

    # Loop over features
    for feat_idx in range(n_feat):
        print(f"\n=== Processing Feature {feat_idx + 1}/{n_feat} ===")

        for time_idx in range(n_time):
            rdm_matrix = np.zeros((n_cond, n_cond))  # Initialize RDM matrix

            start_time = time.time()
            for cond1 in range(n_cond):
                for cond2 in range(cond1 + 1, n_cond):
                    trials1 = power_cue[cond1, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)
                    trials2 = power_cue[cond2, :, :, feat_idx, time_idx]  # Shape: (n_trial, n_chan)

                    X = np.vstack((trials1, trials2))
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    y = np.array([0] * n_trial + [1] * n_trial)  # Labels for SVM

                    # Parallel SVM training (n_permutations)
                    # svm_results = Parallel(n_jobs=-1)(
                    #     delayed(train_svm_permutation)(X, y) for _ in range(n_permutations)
                    # )
                    svm_results = []
                    for _ in range(n_permutations):
                        svm_result = train_svm_permutation(X, y)
                        svm_results.append(svm_result)

                    # Extract weights and accuracies
                    svm_weights, svm_accuracies = zip(*svm_results)
                    svm_weights = np.array(svm_weights)  # Shape: (n_permutations, n_chan)
                    svm_accuracies = np.array(svm_accuracies)  # Shape: (n_permutations,)

                    # Store SVM weights and accuracies
                    all_svm_weights[feat_idx, :, :] = svm_weights.T  # Store per channel
                    all_svm_accuracies[feat_idx, :] = svm_accuracies

                    # Compute mean accuracy for RDM
                    mean_acc = np.mean(svm_accuracies)
                    rdm_matrix[cond1, cond2] = mean_acc
                    rdm_matrix[cond2, cond1] = mean_acc  # Symmetric matrix

            process_time = time.time() - start_time
            print(f"Time taken: {process_time:.2f}s")

            # Save RDM
            np.save(out_p / f"{sub_str}_feature-{feat_idx+1}_rdm.npy", rdm_matrix)

        # Save SVM Weights & Accuracy
        np.save(out_p / f"{sub_str}_feature-{feat_idx+1}_svm_weights.npy", all_svm_weights[feat_idx])
        np.save(out_p / f"{sub_str}_feature-{feat_idx+1}_svm_accuracies.npy", all_svm_accuracies[feat_idx])

    print(f"\nAll RDMs, SVM weights, and accuracies saved for subject {sub_str}!\n")