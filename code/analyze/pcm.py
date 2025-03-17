from pathlib import Path
import PcmPy as pcm
import numpy as np
from scipy.spatial.distance import squareform

import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
subjects = [2]  # Add as many subject IDs as you need
task = "craa"
desc = "Apply morlet wavlet transform."

base_dir = Path("..") / "data" / "derivatives"
in_folder = "cwt"
# out_folder = "cwt"

epoch_boundary_cue = [-1.3, 1]  # sec

for sub_id in subjects:
    sub_str = f"sub-{sub_id:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")
    in_p = base_dir / in_folder / sub_str / "eeg"
    in_file_cue = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-cue.npy"
    in_file_target = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-target.npy"

    power_cue = np.load(in_file_cue)
    power_cue *= 10e11  # V -> uV

    # compare with winko's data
    # mat_data = loadmat('/Users/jinhee/Library/CloudStorage/SynologyDrive-jinhee20250212/[]WORK-CMU_PhD/Code/AuditoryAttentionRSAWinko/Data/6_condData_CSD_v1/Cue_RSAS_PLT2_Combined_v6.mat')
    
    n_cond, n_trial, n_chan, n_feat, n_time = power_cue.shape
    cond_vec = np.repeat(np.arange(n_cond), n_trial)
    # cond_vec = np.arange(n_cond)
    category_labels = np.array([0] * 8 + [1] * 6 + [2] * 7)  # 0: space attention, 1: talker attention, 2: no attention (relax)
    
    quarter_trials = n_trial // 2  # TODO: check that it's divided into 25:75 or 75:25
    partitions = np.zeros(n_trial, dtype=int)
    partitions[:quarter_trials] = 1
    part_vec = np.tile(partitions, n_cond)
    # part_vec = partitions
    
    G_space = np.zeros((n_cond, n_cond))
    G_talker = np.zeros((n_cond, n_cond))
    G_relax = np.zeros((n_cond, n_cond))
    for i in range(n_cond):
        for j in range(n_cond):
            category = category_labels[i]
            if category_labels[i] == category_labels[j]:
                if category == 0:
                    G_space[i, j] = 1
                elif category == 1:
                    G_talker[i, j] = 1                    
                elif category == 2:
                    G_relax[i, j] = 1    
                    
    attend_vec = np.where(category_labels < 2, 1, 0)  # 1 for attention, 0 for relax
    G_attend = np.outer(attend_vec, attend_vec)
    
    space_vec = np.where(category_labels == 0, 1, 0)
    G_space_only = np.outer(space_vec, space_vec)
    talker_vec = np.where(category_labels == 1, 1, 0)
    G_talker_only = np.outer(talker_vec, talker_vec)
    
    models = []
    
    G_independent = np.eye(n_cond)
    models.append(pcm.FixedModel("Independent", G_independent))
    models.append(pcm.ComponentModel("3-category", [G_space, G_talker, G_relax]))
    models.append(pcm.ComponentModel("attend-relax", [G_attend]))
    models.append(pcm.ComponentModel("combined-2x2", [G_attend, G_space, G_talker]))
    
    log_likelihoods = {m.name: [] for m in models}
    rdm_list = []
    
    # for f in range(n_feat):
    feat_i = 2  # alpha
    for t in range(n_time):
        print(f"[Time {t}]")
        data_t = power_cue[:, :, :, feat_i, t]
        obs_t = data_t.reshape(-1, n_chan)  # flatten trials
        # obs_t = data_t.mean(axis=1)  # mean trials
        
        dataset_t = pcm.dataset.Dataset(obs_t, obs_descriptors={"cond_vec": cond_vec, "part_vec": part_vec})
        
        T, theta = pcm.inference.fit_model_individ([dataset_t], models, fit_scale=True, noise_cov=None)  # TODO: check fit_noise
        
        for m_i, m in enumerate(models):
            ll = T.likelihood[m.name][0]
            log_likelihoods[m.name].append(ll)
            G_pred, _ = m.predict(theta[m_i][:m.n_param])
            plt.imshow(G_pred)
            plt.show()
            
        # G,_ = M[4].predict(theta_gr[4][:M[4].n_param])
        # plt.imshow(G)
        
        # dissimilarity based on G estimate      
        G_hat, _ = pcm.est_G_crossval(obs_t, cond_vec, part_vec)
        C = pcm.pairwise_contrast(np.arange(n_cond))
        rdm = squareform(np.diag(C @ G_hat @ C.T))
        
        # rdm based on correlation
        # pattern_avg = obs_t.reshape(n_cond, n_trial, n_chan).mean(axis=1)  # shape: n_cond x n_chan
        # corr_matrix = np.corrcoef(pattern_avg)  # correlation between patterns (default treats rows as variables)
        # rdm_corr = 1 - corr_matrix  # convert similarity to dissimilarity
        # np.fill_diagonal(rdm_corr, 0)  # ensure exact 0s on diagonal
        rdm_list.append(rdm)
        

    time_vec = np.linspace(epoch_boundary_cue[0], epoch_boundary_cue[1], n_time)
    target_times = np.array([-1.1, -0.5, 0, 0.25, 0.5, 1])
    half_window = 12  # 12 before + 12 after + 1 center = 25 total
    indices = np.array([np.argmin(np.abs(time_vec - t)) for t in target_times])

    target_time_RDMs = []
    for i, target_idx in enumerate(indices):
        start_idx = max(target_idx - half_window, 0)
        end_idx = min(target_idx + half_window + 1, len(rdm_list))  # +1 to include 25 elements
        
        avg_RDM = np.mean(rdm_list[start_idx:end_idx], axis=0)
        target_time_RDMs.append(avg_RDM)
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(rdm, cmap="viridis", square=True)
        plt.title(f"PCM-based RDM at time {target_times[i]}")
        plt.tight_layout()
        plt.savefig(in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-cue_rdm-{target_times[i]}s.png")
        plt.close()
        
    target_time_RDMs = np.array(target_time_RDMs)
    breakpoint()
     
