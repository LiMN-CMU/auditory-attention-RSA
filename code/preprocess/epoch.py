import argparse
from pathlib import Path
import mne
import numpy as np
import json
from mne_bids import BIDSPath

# Parameters
def run(config, sub_i, mode="manual"):
    task = config["task"]
    subjects = config["subjects"]

    base_dir = Path(config["base_dir"])
    config_epoch = config["epoch"]
    in_dir = base_dir / config_epoch["input_folder"]
    out_dir = base_dir / config_epoch["output_folder"]
    
    epoch_boundary_cue = config_epoch["epoch_boundary_cue"]  # sec
    epoch_boundary_target = config_epoch["epoch_boundary_target"]  # sec
    baseline_boundary = config_epoch["baseline_boundary"]  # 500~200ms before the visual cue 
    # amplitude_thres = 400e-6  # 400 uV

    event_dict_fpath = base_dir / "etc" / "event_id.json"
    with open(event_dict_fpath, 'r', encoding='utf-8') as jf:
        event_dict = json.load(jf)

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
    
    bids_out_cue = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        processing=out_dir.name,
        datatype="eeg",
        root=out_dir,
        description="cue"
    )
    bids_out_target = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        processing=out_dir.name,
        datatype="eeg",
        root=out_dir,
        description="target"
    )
    out_file_cue = bids_out_cue.fpath
    out_file_target = bids_out_target.fpath
    out_file_cue.parent.mkdir(parents=True, exist_ok=True)  

    print(f"Loading file: {in_file}")
    raw = mne.io.read_raw_fif(in_file, preload=True)
    events = mne.find_events(raw, shortest_event=1, mask=255, mask_type="and")

    # Convert events to annotations
    annotations = mne.annotations_from_events(
        events=events,
        event_desc={val: key for key, val in event_dict.items()},  # reversed dictionary
        sfreq=raw.info["sfreq"]
    )
    raw.set_annotations(annotations)    
    
    # Filter events (only condition cues)
    event_dict_baseline = {key: val for key, val in event_dict.items() if "condition" in key}  # condition cues
    event_dict_cue = {key: val for key, val in event_dict.items() if "stim/type/cue" in key}  # audio cues
    event_dict_target = {key: val for key, val in event_dict.items() if "stim/type/target" in key}  # target syllable onset

    epochs_baseline = mne.Epochs(
        raw, events, event_id=event_dict_baseline, tmin=baseline_boundary[0], tmax=baseline_boundary[1],
        baseline=None, reject_by_annotation=True, preload=True, reject=None
    )
    eeg_baseline = epochs_baseline.crop(tmin=baseline_boundary[0], tmax=baseline_boundary[1]).pick("eeg").get_data()
    eeg_baseline_mean = eeg_baseline.mean(axis=2, keepdims=True)
    condition_labels = epochs_baseline.events[:, 2]
    
    epochs_cue = mne.Epochs(
        raw, events, event_id=event_dict_cue, tmin=epoch_boundary_cue[0], tmax=epoch_boundary_cue[1],
        baseline=None, reject_by_annotation=True, preload=True, reject=None, picks="eeg"
    )
    epochs_target = mne.Epochs(
        raw, events, event_id=event_dict_target, tmin=epoch_boundary_target[0], tmax=epoch_boundary_target[1],
        baseline=None, reject_by_annotation=True, preload=True, reject=None, picks="eeg"
    )
    # baseline correction
    epochs_cue._data -= eeg_baseline_mean
    epochs_target._data -= eeg_baseline_mean
    
    if mode == "manual":
        epochs_cue.plot(events=epochs_cue.events, show=True)
        epochs_target.plot(events=epochs_target.events, show=True)

    # save file and full event data
    epochs_cue.save(out_file_cue.with_suffix(".fif"), overwrite=True)
    epochs_target.save(out_file_target.with_suffix(".fif"), overwrite=True)
    label_fpath = out_file_cue.parent / (out_file_cue.stem.split("_desc")[0] + '_conditions.npy')
    np.save(label_fpath, condition_labels)
    
    # event_file = out_file.with_suffix('.eve')
    # mne.write_events(event_file, events, overwrite=True)
    
    
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="preprocessing-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        run(config, sub_i)