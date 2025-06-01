import json
from pathlib import Path
import mne
from mne_bids import BIDSPath
from mne_icalabel import label_components

def choose_n_components(ica, threshold=0.99):
    """Return number of ICA components that explain the given cumulative variance."""
    explained_var = ica.pca_explained_variance_
    cumulative_var = explained_var.cumsum()
    print(cumulative_var)
    plateau = cumulative_var[-1]
    scaled_threshold = threshold * plateau
    n_components = (cumulative_var < scaled_threshold).sum() + 1
    return n_components

def apply_top_components(config, sub_i, mode="manual"):
    task = config["task"]
    sub_str = f"sub-{sub_i:03d}"

    base_dir = Path(config["base_dir"])
    config_ICA = config["select_top_ICA"]
    in_dir = base_dir / config_ICA["input_folder"]
    out_dir = base_dir / config_ICA["output_folder"]

    iclabel_component_thres = config_ICA["iclabel_component_threshold"]
    iclabel_prob_thres = config_ICA["iclabel_proability_threshold"]
    iclabel_bad_class = config_ICA["iclabel_bad_class"]
    variance_threshold = config_ICA["explained_variance_thres"]
    
    bids_in = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        suffix='eeg',
        processing=in_dir.name,
        extension='.fif',
        datatype='eeg',
        root=in_dir
    )
    in_file = bids_in.fpath
    ica_fname = in_file.parent / f"{in_file.stem}-ica.fif"
    
    bids_out = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        suffix='eeg',
        processing=out_dir.name,
        extension='.fif',
        datatype='eeg',
        root=out_dir
    )
    out_file = bids_out.fpath
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Subject: {sub_str} ===")
    print(f"Loading: {in_file}")
    raw = mne.io.read_raw_fif(in_file, preload=True)
    ica = mne.preprocessing.read_ica(ica_fname)

    # Run ICLabel
    # Step 1: Identify components to exclude based on ICLabel
    iclabel_result = label_components(raw, ica, method="iclabel")
    ic_labels = iclabel_result["labels"]
    ic_label_probs = iclabel_result["y_pred_proba"]
    artifact_idx = [
        comp_i for comp_i, (label, prob) in enumerate(zip(ic_labels, ic_label_probs))
        if comp_i <= iclabel_component_thres and prob > iclabel_prob_thres and label in iclabel_bad_class
    ]
    
    # # Plot and save ICA properties
    print(f"ICLabel Result: {ic_labels}")
    print(f"ICLabel Result: {ic_label_probs}")
    print(f"Bad Component Indices: {artifact_idx}")
    if artifact_idx:
        if mode == "manual":
            fig1 = ica.plot_components(picks=range(0, iclabel_component_thres), show=True)
            fig1.savefig(out_file.parent / f"{out_file.stem}_ica_components.png", dpi=300)
            fig2 = ica.plot_properties(raw, picks=artifact_idx, verbose=False, show=True)
            for bad_i, f in zip(artifact_idx, fig2):
                f.savefig(out_file.parent / f"{out_file.stem}_bad_ica_property_{bad_i}.png", dpi=300)
        elif mode == "auto":
            fig1 = ica.plot_components(picks=range(0, iclabel_component_thres), show=False)
            fig1.savefig(out_file.parent / f"{out_file.stem}_ica_components.png", dpi=300)
            fig2 = ica.plot_properties(raw, picks=artifact_idx, verbose=False, show=False)
            for bad_i, f in zip(artifact_idx, fig2):
                f.savefig(out_file.parent / f"{out_file.stem}_bad_ica_property_{bad_i}.png", dpi=300)

    # Compute variance and keep only best components (after excluding artifacts)
    n_keep = choose_n_components(ica, threshold=variance_threshold)
    variance_keep_idx = list(range(n_keep))

    # Final exclusion list = all components - (artifact ∪ variance_kept)
    all_components = set(range(ica.n_components_))
    keep_components = set(variance_keep_idx)
    bad_components = set(artifact_idx)
    exclude = list(all_components - (keep_components - bad_components))

    print(f"  - Artifact components: {sorted(artifact_idx)}")
    print(f"  - Keeping components (variance): {sorted(variance_keep_idx)}")
    print(f"  - Final excluded components: {sorted(exclude)}")

    # # Apply ICA with full exclusion
    ica.apply(raw, exclude=exclude)
    raw.interpolate_bads(reset_bads=True)

    # Save cleaned raw
    raw.save(out_file.with_suffix(".fif"), overwrite=True)

    # Save metadata
    ica_result_meta_dict = {
        "BadComponents": list(sorted(bad_components)),
        "ICAKeptComponents": list(sorted(variance_keep_idx)),
        "ExcludedComponents": list(sorted(exclude)),
        "ExplainedVarianceThreshold": variance_threshold
    }
    json_outfile = out_file.with_suffix(".json")
    with open(json_outfile, "w") as f:
        json.dump(ica_result_meta_dict, f, indent=4)
    
    print(f"  - Saved cleaned EEG to {out_file}")
    print(f"  - Saved metadata to {json_outfile}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="preprocessing-001", help="Configuration ID")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{args.config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    for sub_i in config["subjects"]:
        apply_top_components(config, sub_i, mode=config["mode"])