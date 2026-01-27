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
from sklearn.preprocessing import StandardScaler

# --- colorblind-safe palette ---
okabe_ito = [
    # "#E14949",
    # "#91C95A",
    "#5BB5E5",
    "#F5A027",
    "#606563"
]

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

def average_by_category(data, category_num_dict={"space": 8, "talker": 6, "relax": 7}):
    """
    Average between/within category values for time-resolved RDMs or weight matrices,
    now supporting an additional leading subject axis.

    Parameters
    ----------
    data : np.ndarray
        Shape:
            (n_subject, timepoints, n_cond, n_cond)                  # RDMs
            (n_subject, timepoints, n_cond, n_cond, n_channels)      # weights
        Must be symmetric with zeros on the diagonal (RDM case).

    category_num_dict : dict
        Keys: category names
        Values: number of conditions in each category.

    Returns
    -------
    results : dict
        Keys: 'catA-catB'
        Values:
            Shape (n_subject, timepoints, n_channels) if weights
            Shape (n_subject, timepoints) if RDMs
    """
    ndim = data.ndim
    if ndim == 4:
        n_subj, timepoints, n_cond, _ = data.shape
        n_channels = None
    elif ndim == 5:
        n_subj, timepoints, n_cond, _, n_channels = data.shape
    else:
        raise ValueError(f"Data must have shape (subj, time, cond, cond) or (subj, time, cond, cond, channels), got {data.shape}")

    assert n_cond == sum(category_num_dict.values()), \
        f"Mismatch: {n_cond} conditions vs {sum(category_num_dict.values())} from category counts"

    # Condition indices for each category
    cat_labels = np.concatenate([[i] * n for i, n in enumerate(category_num_dict.values())])
    cat_indices = [np.where(cat_labels == i)[0] for i in range(len(category_num_dict))]
    cat_names = list(category_num_dict.keys())

    def extract_and_average(cond_idx1, cond_idx2, symmetric=True):
        mask = np.zeros((n_cond, n_cond), dtype=bool)
        for i in cond_idx1:
            for j in cond_idx2:
                if i != j and (not symmetric or i < j):
                    mask[i, j] = True
        if n_channels is None:
            # (n_subj, time, num_pairs)
            return data[:, :, mask]
        else:
            # (n_subj, time, num_pairs, n_channels)
            return data[:, :, mask, :]

    results = {}

    # Within-category
    if ndim == 4:  # only for RDMs
        for i in range(len(cat_names)):
            key = f"{cat_names[i]}-{cat_names[i]}"
            vals = extract_and_average(cat_indices[i], cat_indices[i])
            results[key] = vals.mean(axis=2)  # average across pairs

    # Between-category
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            key = f"{cat_names[i]}-{cat_names[j]}"
            vals = extract_and_average(cat_indices[i], cat_indices[j], symmetric=False)
            results[key] = vals.mean(axis=2)  # average across pairs

    return results

def category_dissimilarity_index(avg_results, category_names, time_indices, ci=0.95):
    """
    Compute category dissimilarity index (per timepoint) with mean ± margin of error
    across subjects.

    Parameters
    ----------
    avg_results : dict
        Output of `average_by_category`, where each value has shape (n_sub, n_timepoints).

    category_names : list of str
        Names of the categories, e.g. ["space", "talker", "relax"].

    time_indices : tuple (start_i, end_i)
        Time window indices to analyze. Only this window will be used.

    ci : float, optional
        Confidence interval level (default=0.95).

    Returns
    -------
    sim_results : dict
        Keys: 'catA-catB'
        Values: dict with
            - 'values': array (n_sub, n_time_window)
                Raw dissimilarity values per subject per timepoint.
            - 'mean': array (n_time_window,)
                Mean across subjects for each timepoint.
            - 'margin_error': array (n_time_window,)
                Margin of error for the CI at each timepoint.
    """
    start_i, end_i = time_indices
    sim_results = {}
    alpha = 1 - ci

    for i in range(len(category_names)):
        for j in range(i + 1, len(category_names)):
            between_key = f"{category_names[i]}-{category_names[j]}"
            within_A = avg_results[f"{category_names[i]}-{category_names[i]}"][:, start_i: end_i]
            within_B = avg_results[f"{category_names[j]}-{category_names[j]}"][:, start_i: end_i]
            between_AB = avg_results[between_key][:, start_i: end_i]
            
            # (n_sub, n_time_window)
            sim_index_vals = between_AB - 0.5 * (within_A + within_B)
            sim_index_vals = sim_index_vals.mean(axis=1)  # mean across the time window

            # Compute stats across subjects, for each timepoint
            n_sub = sim_index_vals.shape[0]
            mean_val = sim_index_vals.mean(axis=0)
            sem_val = stats.sem(sim_index_vals, axis=0)
            t_crit = stats.t.ppf(1 - alpha/2, n_sub - 1)
            margin_error = t_crit * sem_val

            sim_results[between_key] = {
                "values": sim_index_vals,      # shape (n_sub, n_time_window)
                "mean": mean_val,           
                "margin_error": margin_error
            }
            
            # sim_results[between_key] = sim_index_vals.mean()

    return sim_results

def holm_reject(pvals, alpha=0.05):
        """Holm-Bonferroni rejection decisions + adjusted p-values."""
        pvals = np.asarray(pvals)
        m = pvals.size
        order = np.argsort(pvals)
        p_sorted = pvals[order]

        # step-down thresholds
        thresh = alpha / (m - np.arange(m))
        reject_sorted = np.zeros(m, dtype=bool)

        for i in range(m):
            if p_sorted[i] <= thresh[i]:
                reject_sorted[i] = True
            else:
                break  # once one fails, all larger p fail

        # Holm adjusted p-values (step-down)
        p_adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p_sorted)
        p_adj_sorted = np.clip(p_adj_sorted, 0, 1)

        reject = np.zeros(m, dtype=bool)
        p_adj = np.zeros(m, dtype=float)
        reject[order] = reject_sorted
        p_adj[order] = p_adj_sorted
        return reject, p_adj

def run_spatiotemporal_cluster_permutation_tests(category_svm_weights, info, ch_type="eeg",
                                  n_permutations=1000, alpha=0.05, seed=42):
    """
    Run cluster-based permutation tests (TFCE) across all category SVM weights.

    Parameters
    ----------
    category_svm_weights : dict
        Dict of {category_name: np.ndarray (n_subjects, n_time, n_channels)}.
    info : mne.Info
        MNE info structure for adjacency.
    ch_type : str
        Channel type for adjacency (default 'eeg').
    n_permutations : int
        Number of permutations.
    alpha : float
        Base alpha level before correction.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing TFCE maps, clusters, p-values, etc.
    """
    
    rng = np.random.RandomState(seed)
    # ch_adj, _ = find_ch_adjacency(info, ch_type=ch_type)
    adj, adj_names = mne.channels.read_ch_adjacency("biosemi64")
    keep = [ch for ch in info.ch_names if ch in adj_names]
    idx = [adj_names.index(ch) for ch in keep]
    adj = adj[idx][:, idx]  # reorder/subselect)

    results = {
        "per_cond": {},
        "all_cluster_ps": []
    }

    for cat_name, cat_val in category_svm_weights.items():
        n_sub, n_time, n_chan = cat_val.shape
        
        T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
            cat_val,
            threshold=None,        # TFCE
            tail=0,
            n_permutations=n_permutations,
            adjacency=adj,
            out_type="mask",
            n_jobs=1,
            seed=rng
        )

        rec = {
            "T_obs": T_obs.reshape(n_time, n_chan),
            "clusters": clusters,
            "clusters_2d": [c.reshape(n_time, n_chan) for c in clusters],
            "cluster_p_values": cluster_p
        }
        results["per_cond"][cat_name] = rec
        results["all_cluster_ps"].extend(cluster_p)
        
    contrast_names = list(results["per_cond"].keys())
    p_contrast = []
    for name in contrast_names:
        p = results["per_cond"][name]["cluster_p_values"]
        p_contrast.append(p.min() if len(p) else 1.0)
    p_contrast = np.array(p_contrast)
    
    reject, p_contrast_adj = holm_reject(p_contrast, alpha=alpha)
    results["contrast_names"] = contrast_names
    results["p_contrast_min_cluster"] = p_contrast
    results["p_contrast_holm"] = p_contrast_adj
    results["reject_holm"] = reject
    
    return results

def run_temporal_cluster_permutation_tests(category_sim_indices,
                                  n_permutations=1000, alpha=0.05, n_correction=6, seed=42):
    """
    Run cluster-based permutation tests (TFCE) across all category SVM weights.

    Parameters
    ----------
    category_sim_indices : dict
        Dict of {category_name: np.ndarray (n_subjects, n_time)}.
    n_permutations : int
        Number of permutations.
    alpha : float
        Base alpha level before correction.
    n_correction : int
        Number of comparisons for Bonferroni correction.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing TFCE maps, clusters, p-values, etc.
    """
    alpha_corrected = alpha / n_correction
    rng = np.random.RandomState(seed)

    results = {
        "per_cond": {},
        "all_cluster_ps": []
    }

    for cat_name, cat_val in category_sim_indices.items():
        sim_indices = np.stack(cat_val["values"], axis=1)
        n_sub, n_time = sim_indices.shape

        T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
            sim_indices,
            threshold=None,        # TFCE
            tail=0,
            n_permutations=n_permutations,
            out_type="mask",
            n_jobs=1,
            seed=rng
        )

        rec = {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p
        }
        results["per_cond"][cat_name] = rec
        results["all_cluster_ps"].extend(cluster_p)

    contrast_names = list(results["per_cond"].keys())
    p_contrast = []
    for name in contrast_names:
        p = results["per_cond"][name]["cluster_p_values"]
        p_contrast.append(p.min() if len(p) else 1.0)
    p_contrast = np.array(p_contrast)
    
    reject, p_contrast_adj = holm_reject(p_contrast, alpha=alpha)
    results["contrast_names"] = contrast_names
    results["p_contrast_min_cluster"] = p_contrast
    results["p_contrast_holm"] = p_contrast_adj
    results["reject_holm"] = reject
    results["all_cluster_ps"] = np.array(results["all_cluster_ps"])
    
    return results

def window_cluster_mask(results, category_name, time_idx, alpha=0.05, which="within", ylim=(None, None)):
    """
    Build a boolean mask over channels indicating membership in any
    significant spatiotemporal cluster that overlaps [t_start:t_end).

    results: dict produced earlier (with 'per_cond' entries)
    category_name
    alpha: significance threshold
    Returns: mask_ch (n_chan,) boolean
    """
    # Holm gate across contrasts
    if "reject_holm" in results and "contrast_names" in results:
        j = results["contrast_names"].index(category_name)
        if not results["reject_holm"][j]:
            return None  # contrast not significant after Holm

    # within-contrast cluster selection (TFCE-corrected already)
    rec = results["per_cond"][category_name]
    pvals = rec["cluster_p_values"]

    sig_idxs = [k for k, p in enumerate(pvals) if p <= alpha]
    if not sig_idxs:
        return None

    # Union across all significant clusters that touch any time in the window
    n_time, n_chan = rec["clusters_2d"][0].shape
    mask_ch = np.zeros(n_chan, dtype=bool)
    for k in sig_idxs:
        cl = rec["clusters_2d"][k]  # shape (n_time, n_chan), bool
        mask_ch |= cl[time_idx, :]  # directly take this row

    if not mask_ch.any():
        # print(f"No cluster touches time {time_idx} for {category_name}")
        return None
    
    print(f"Cluster touches time {time_idx} for {category_name}")
    return mask_ch

def plot_topomap(
    *,
    evoked=None,
    weights=None,
    info=None,
    mask_ch=None,
    config_rsa=None,
    epoch_type=None,
    figsize=(8, 8),
    times=0,
    size=3,
    colorbar=False,
    mask_params=None,
    show=False,
    cmap="PRGn"):
    """
    Unified topomap plotting function.

    You can provide either:
      (A) evoked=mne.Evoked
    or
      (B) weights=(n_channels,) and info=mne.Info  -> will build EvokedArray

    Optionally overlay a channel mask (mask_ch) as markers.

    Parameters
    ----------
    evoked : mne.Evoked or None
        Evoked object to plot.
    weights : np.ndarray or None
        1D array (n_channels,) used to create an EvokedArray if evoked is None.
    info : mne.Info or None
        Required if weights is provided and evoked is None.
    mask_ch : np.ndarray or None
        Boolean array (n_channels,) marking significant channels.
    config_rsa : dict
        Must contain f"weight_boundary_{epoch_type}" -> (min, max).
    epoch_type : str
        Used to index weight boundaries.
    figsize : tuple
        Figure size.
    times : float | int
        Time(s) to plot in seconds or sample index depending on evoked; here it’s 0 by your usage.
    size : float
        Topomap size passed to MNE.
    colorbar : bool
        Whether to show colorbar.
    mask_params : dict or None
        Passed to MNE plot_topomap. If None, a sensible default is used.
    show : bool
        Whether to display immediately.

    Returns
    -------
    fig, ax
    """
    if config_rsa is None or epoch_type is None:
        raise ValueError("config_rsa and epoch_type must be provided.")

    weight_min, weight_max = config_rsa[f"weight_boundary_{epoch_type}"]

    # Build evoked if needed
    if evoked is None:
        if weights is None or info is None:
            raise ValueError("Provide either evoked=... OR (weights=... AND info=...).")
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D (n_channels,), got shape {weights.shape}")
        evoked = mne.EvokedArray(weights[:, np.newaxis], info)

    # Prepare mask
    mask = None
    if mask_ch is not None:
        mask_ch = np.asarray(mask_ch)
        if mask_ch.dtype != bool:
            mask_ch = mask_ch.astype(bool)
        # MNE expects mask shape (n_channels, n_times) for Evoked topomap
        mask = mask_ch[:, np.newaxis]

    if mask_params is None:
        mask_params = dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=8,
        )

    fig, ax = plt.subplots(figsize=figsize)
    evoked.plot_topomap(
        times=times,
        scalings=1,
        vlim=(weight_min, weight_max),
        time_format="",
        size=size,
        colorbar=colorbar,
        mask=mask,
        mask_params=mask_params if mask is not None else None,
        axes=ax,
        show=show,
        cmap=cmap
    )
    return fig, ax

def plot_dissimilarity_index(group_sim_indices_rdms, n_times, config_rsa, epoch_type, figsize=(12, 2)):
    """
    Plot dissimilarity/similarity index over time with confidence intervals.

    Parameters
    ----------
    group_sim_indices_rdms : dict
        Dictionary with category names as keys and dicts containing 'mean' and 'margin_error'.
    target_times : list or np.ndarray
        Time points corresponding to the x-axis.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes._axes.Axes
        The axes object containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_min, y_max = config_rsa[f"similarity_index_boundary_{epoch_type}"]
    start_time, end_time = config_rsa[f"epoch_boundary_{epoch_type}"]

    x_positions = np.linspace(start_time, end_time, n_times)

    for i, (cat_names, vals) in enumerate(group_sim_indices_rdms.items()):
        color = okabe_ito[i % len(okabe_ito)]
        mean = np.array(vals["mean"])
        margin = np.array(vals["margin_error"])

        # Plot the mean line
        # ax.plot(x_positions, mean, marker='o', label=cat_names, color=color)
        ax.plot(x_positions, mean, label=cat_names, color=color)

        # Add CI shading (mean ± margin)
        ax.fill_between(
            x_positions,
            mean - margin,
            mean + margin,
            color=color,
            alpha=0.2
        )

    # plt.xlabel("Time (ms)")
    # plt.xticks(x_positions, labels=[t * 1000 for t in target_times])
    if epoch_type == "cue":
        # plt.xticks([-1.1, -0.5, 0, 0.15, 0.25, 1], ["-1100ms", "-500ms", "0ms", "150ms", "250ms", "1000ms"])
        plt.xticks([-1, -0.5, 0, 0.5, 1], ["-1000ms", "-500ms", "0ms", "500ms", "1000ms"])
    elif epoch_type == "target":
        # plt.xticks([-0.2, -0.1, 0, 0.15, 0.3, 0.4], ["-200ms", "-100ms", "0ms", "150ms", "300ms", "400ms"])
        plt.xticks([-0.4, 0, 0.5], ["-400ms", "0ms", "500ms"])
    # plt.ylabel("Similarity Index")
    # ax.set_xticks([])          # remove tick positions
    # ax.set_xticklabels([])     # remove tick labels
    # ax.spines['bottom'].set_visible(False)  # hide x-axis line
    ax.set_xlabel("")          # remove x-axis label

    ax.set_ylim((y_min, y_max))  # Example: 0–1 range

    # Apply new legend order
    # handles, labels = ax.get_legend_handles_labels()
    # order = [1, 2, 0]
    # ax.legend([handles[idx] for idx in order],
    #           [labels[idx] for idx in order],
    #           loc="upper left")
    ax.legend()

    fig.tight_layout()
    return fig, ax

def plot_significant_clusters(ax, x_positions, results, cluster_alpha=0.05, y_step=0.04, lw=1.5):
    x_positions = np.asarray(x_positions)
    n_times = len(x_positions)

    contrast_names = results.get("contrast_names", list(results["per_cond"].keys()))
    reject_holm = results.get("reject_holm", None)

    trans = ax.get_xaxis_transform()  # x=data, y=axes coords
    y0 = 1.02

    for i, cat_name in enumerate(contrast_names):
        if reject_holm is not None and not reject_holm[i]:
            continue

        rec = results["per_cond"][cat_name]
        color = okabe_ito[i % len(okabe_ito)]
        y = y0 + i * y_step

        for cl, pval in zip(rec["clusters"], rec["cluster_p_values"]):
            if pval > cluster_alpha:
                continue

            # convert cluster -> time indices
            if isinstance(cl, tuple):
                c0 = cl[0]
                if isinstance(c0, slice):
                    t_inds = np.arange(n_times)[c0]
                else:
                    c0 = np.asarray(c0)
                    t_inds = np.flatnonzero(c0) if c0.dtype == bool else c0.astype(int)
            else:
                cl = np.asarray(cl)
                if cl.dtype == bool and cl.ndim == 1:
                    t_inds = np.flatnonzero(cl)
                elif cl.dtype == bool and cl.ndim >= 2:
                    t_inds = np.flatnonzero(cl.reshape(n_times, -1).any(axis=1))
                else:
                    t_inds = np.asarray(cl, int)

            if t_inds.size == 0:
                continue

            t_inds = np.unique(np.sort(t_inds))
            breaks = np.where(np.diff(t_inds) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, t_inds.size - 1]

            for s, e in zip(starts, ends):
                ax.hlines(y, x_positions[t_inds[s]], x_positions[t_inds[e]],
                          color=color, lw=lw, transform=trans, clip_on=False)

def plot_rdms(target_rdm_spaced, acc_min, acc_max, figsize=(8, 8), cmap="viridis"):
    """
    Plot a Representational Dissimilarity Matrix (RDM) heatmap.

    Parameters
    ----------
    target_rdm_spaced : np.ndarray
        2D array representing the RDM.
    acc_min : float
        Minimum value for color scaling.
    acc_max : float
        Maximum value for color scaling.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes._axes.Axes
        The axes object containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        target_rdm_spaced,
        cmap=cmap,
        square=True,
        cbar=False,
        vmin=acc_min,
        vmax=acc_max,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    # plt.title(f"PCM-based RDM at time {target_times[time_i]} sec")
    fig.tight_layout()

    return fig, ax

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
    montage = mne.channels.read_custom_montage(montage_p / "chanlocs_64_3_eye_chan.locs")
    ch_names = montage.ch_names[:64]  # Ensure matching length  TODO: n_channel hardcoded
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * 64)
    info.set_montage(montage, on_missing="raise")

    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        print(f"=== Loading {epoch_type} epochs ===")
        acc_min = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][0] 
        acc_max = config_rsa[f"rdm_accuracy_boundary_{epoch_type}"][1]

        all_rdms = np.load(in_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm.npy")
        all_svm_weights = np.load(in_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights.npy")
        
        print(all_rdms.shape)
        print(all_svm_weights.shape)
        
        n_sub, n_time, n_cond, _ = all_rdms.shape
        time_vec = np.linspace(epoch_boundary[0], epoch_boundary[1], n_time)
        
        plot_target_times = config_rsa[f"plot_target_time_{epoch_type}"]
        plot_target_time_indices = [np.argmin(np.abs(time_vec - t)) for t in plot_target_times]
        
        # average across subjects
        group_avg_rdms = np.mean(all_rdms, axis=0)  # shape: (n_time, n_cond, n_cond)
        
        # average by task category 
        all_category_svm_weights = average_by_category(all_svm_weights, category_num_dict=category_num_dict)
        group_category_svm_weights = dict()
        for cat_names, cat_vals in all_category_svm_weights.items():
            group_category_svm_weights[cat_names] = np.mean(cat_vals, axis=0)  # average across subjects
        
        # Target time analysis
        for target_time, target_time_idx in zip(plot_target_times, plot_target_time_indices):
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            window_rdms = group_avg_rdms[window_start:window_end]
            target_rdm = np.mean(window_rdms, axis=0)

            # 1. Plot RDMs
            # Add visual spacing between condition groups
            target_rdm_spaced = add_spacing(target_rdm, breaks)
            fig, ax = plot_rdms(target_rdm_spaced, acc_min, acc_max)
            out_file = out_dir / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm_{target_time:.1f}s.png"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=300, transparent=True)
            plt.close(fig)
            
            # 2-1. Plot SVM weight importance map
            for cat_name, cat_weights in group_category_svm_weights.items():
                window_svm_weights = cat_weights[window_start:window_end]
                target_svm_weights = np.mean(window_svm_weights, axis=0)
                
                evoked = mne.EvokedArray(target_svm_weights[:, np.newaxis], info)  # Add time dimension
                fig, ax = plot_topomap(
                    evoked=evoked,
                    config_rsa=config_rsa, 
                    epoch_type=epoch_type
                )
                
                fig_name = f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_{target_time:.1f}s_category-{cat_name}"
                fig.savefig(out_file.parent / f"{fig_name}.png", dpi=300, transparent=True)
                plt.close(fig)

        # Whole time analysis
        # average by task category 
        all_category_rdms = average_by_category(all_rdms, category_num_dict=category_num_dict)

        # average across subjects
        group_sim_indices_rdms = defaultdict(lambda: {"values": [], "mean": [], "margin_error": []})
        
        category_svm_weights = defaultdict(list)
        for time_idx in range(len(time_vec)):        
            window_start = max(0, time_idx - window_samples // 2)
            window_end = min(n_time, time_idx + window_samples // 2 + 1)    
            
            # dissimilarity index
            sim_indices_rdms = category_dissimilarity_index(all_category_rdms, list(category_num_dict.keys()), [window_start, window_end])
            for cat_names, cat_values in sim_indices_rdms.items():
                group_sim_indices_rdms[cat_names]["values"].append(cat_values["values"])
                group_sim_indices_rdms[cat_names]["mean"].append(cat_values["mean"])
                group_sim_indices_rdms[cat_names]["margin_error"].append(cat_values["margin_error"])
                
            # for cluster based permutation test
            for cat_name, cat_val in all_category_svm_weights.items():
                target_val = cat_val[:, window_start:window_end, :]  # (n_sub, n_time, n_channel)
                category_svm_weights[cat_name].append(np.mean(target_val, axis=1)) # (n_sub, n_channel)
        for cat_name, cat_val in category_svm_weights.items():
            category_svm_weights[cat_name] = np.stack(cat_val, axis=1)
            
        alpha = config_rsa["significance_alpha"]
        n_perm = config_rsa["significance_n_permutation"]
        n_corr = config_rsa["significance_n_bonferroni_correction"]

        # 2-2. plot masked svm weights map
        cluster based permutation test
        results = run_spatiotemporal_cluster_permutation_tests(
            all_category_svm_weights, info, n_permutations=n_perm, alpha=alpha
            )
        
        for target_time, target_time_idx in zip(plot_target_times, plot_target_time_indices):
            window_start = max(0, target_time_idx - window_samples // 2)
            window_end = min(n_time, target_time_idx + window_samples // 2 + 1)
            
            for cat_name, cat_weights in group_category_svm_weights.items():
                window_svm_weights = cat_weights[window_start:window_end]
                target_svm_weights = np.mean(window_svm_weights, axis=0)
                
                mask_ch = window_cluster_mask(results, cat_name, target_time_idx, alpha=alpha)
                fig, ax = plot_topomap(
                    weights=target_svm_weights,
                    info=info,
                    mask_ch=mask_ch,
                    config_rsa=config_rsa,
                    epoch_type=epoch_type,
                )

                fig_name = (
                    f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_"
                    f"config-{config_id}_haufeweights_{target_time:.1f}s_category-{cat_name}-masked"
                )
                fig.savefig(out_dir / f"{fig_name}.png", dpi=300, transparent=False)
                plt.close(fig)
            
        # # filter meaningful pairs TODO; hard-coded
        # wanted_pairs = ["spaceLeft-spaceRight", "talkerMale-talkerFemale"]
        # selected_group_sim_indices_rdms = {key: group_sim_indices_rdms[key] for key in wanted_pairs if key in group_sim_indices_rdms}
        
        # 3. Plot dissimilarity index        
        results = run_temporal_cluster_permutation_tests(
            group_sim_indices_rdms,
            n_permutations=n_perm,
            alpha=alpha,
            n_correction=n_corr
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
        plot_significant_clusters(ax, x_positions, results)
        out_file = out_dir / "group" / f"group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_sim-index.png"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out_file, dpi=600, transparent=True)
        plt.close(fig)

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