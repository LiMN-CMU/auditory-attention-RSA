import argparse
from pathlib import Path
import json
import mne
from mne_bids import BIDSPath
from mne_icalabel import label_components

# Parameters
def run(config, sub_i, mode="manual"):
    task = config["task"]
    subjects = config["subjects"]

    base_dir = Path(config["base_dir"])
    config_ICA = config["apply_ICA"]
    in_dir = base_dir / config_ICA["input_folder"]
    out_dir = base_dir / config_ICA["output_folder"]
    
    iclabel_component_thres = config_ICA["iclabel_component_threshold"]
    iclabel_prob_thres = config_ICA["iclabel_proability_threshold"]
    iclabel_bad_class = config_ICA["iclabel_bad_class"]

    sub_str = f"sub-{sub_i:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")
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
    ica_fname = in_file.parent / f"{in_file.stem}-ica.fif"  # ICA solution
    
    bids_out = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        processing=out_dir.name,
        extension='.fif',
        datatype='eeg',
        suffix='eeg',
        root=out_dir
    )
    out_file = bids_out.fpath
    out_file.parent.mkdir(parents=True, exist_ok=True) 

    print(f"Loading file: {in_file}")
    raw = mne.io.read_raw_fif(in_file, preload=True)
    ica = mne.preprocessing.read_ica(ica_fname)

    # Run ICLabel
    iclabel_result = label_components(raw, ica, method="iclabel")
    ic_labels = iclabel_result["labels"]
    ic_label_probs = iclabel_result["y_pred_proba"]
    bad_idx = []
    for comp_i, (label, prob) in enumerate(zip(ic_labels, ic_label_probs)):
        if comp_i <= iclabel_component_thres and prob > iclabel_prob_thres and label in iclabel_bad_class:
            bad_idx.append(comp_i)
    # Plot and save ICA properties
    print(f"ICLabel Result: {ic_labels}")
    print(f"ICLabel Result: {ic_label_probs}")
    print(f"Bad Component Indices: {bad_idx}")
    if bad_idx:
        if mode == "manual":
            fig1 = ica.plot_components(picks=range(0, iclabel_component_thres), show=True)
            fig1.savefig(out_file.parent / f"{out_file.stem}_ica_components.png", dpi=300)
            fig2 = ica.plot_properties(raw, picks=bad_idx, verbose=False, show=True)
            for bad_i, f in zip(bad_idx, fig2):
                f.savefig(out_file.parent / f"{out_file.stem}_bad_ica_property_{bad_i}.png", dpi=300)
        elif mode == "auto":
            fig1 = ica.plot_components(picks=range(0, iclabel_component_thres), show=False)
            fig1.savefig(out_file.parent / f"{out_file.stem}_ica_components.png", dpi=300)
            fig2 = ica.plot_properties(raw, picks=bad_idx, verbose=False, show=False)
            for bad_i, f in zip(bad_idx, fig2):
                f.savefig(out_file.parent / f"{out_file.stem}_bad_ica_property_{bad_i}.png", dpi=300)
    
    # remove bad ica components
    ica.apply(raw, exclude=bad_idx)
    
    # interpolated the bad channels after ICA
    raw.interpolate_bads(reset_bads=True)
    
    # Save
    raw.save(out_file.with_suffix(".fif"), overwrite=True)
    
    if bad_idx:
        ica_result_meta_dict = {
            "RejectedComponents": bad_idx,
            "RejectedLabels": [ic_labels[idx] for idx in bad_idx],
            "RejectedLabelProbs": [float(ic_label_probs[idx]) for idx in bad_idx]
        }
        
        # Save ICA rejection metadata as JSON
        json_outfile = out_file.with_suffix(".json")
        with open(json_outfile, "w") as f:
            json.dump(ica_result_meta_dict, f, indent=4)

        print(f"Saved ICA rejection metadata to: {json_outfile}")
        
        
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="preprocessing-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        run(config, sub_i)
