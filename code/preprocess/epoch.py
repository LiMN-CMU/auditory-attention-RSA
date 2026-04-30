import argparse
from pathlib import Path
import mne
import numpy as np
import json
from mne_bids import BIDSPath

def anchor_preceding_multiple(events: np.ndarray,
                              event_codes: list,
                              anchor_code: int = 99):
    """
    Re-anchor multiple event codes to the most recent preceding anchor event.

    Parameters
    ----------
    events : array, shape (n_events, 3)
        Standard MNE events array [sample, 0, code].
    event_codes : list of int
        List of event IDs to re-time (e.g., [52, 53, 54]).
    anchor_code : int
        Code of the onset marker to use as the epoch anchor (e.g., 99).

    Returns
    -------
    new_events : array, shape (m, 3)
        Events array where each row is [anchor_sample, 0, event_code].
        Each event_code is preserved but sample is shifted to the anchor.
    """
    new_events = []
    last_anchor = None

    # ensure int dtype
    events = events.astype(int, copy=False)

    for sample, _, code in events:
        if code == anchor_code:
            last_anchor = sample
        elif code in event_codes:
            if last_anchor is not None:
                new_events.append([last_anchor, 0, code])
            else:
                raise RuntimeError("No anchored events created.")

    return np.array(new_events, dtype=int)


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
    sfreq = raw.info['sfreq']
    
    # Handle subject-specific errors
    if sub_i == 3:
        print("Sub 3")
        start = 2621
        stop = 3506

        # Cut into two halves: before and after
        raw_before = raw.copy().crop(tmin=0, tmax=start, include_tmax=False)
        raw_after = raw.copy().crop(tmin=stop, tmax=None)

        raw = mne.concatenate_raws([raw_before, raw_after])
        
        events = mne.find_events(raw, shortest_event=1, mask=255, mask_type="and")
        
        # Event codes
        COND = np.arange(1, 22)
        CUE = 52
        TARGET = 51
        FEEDBACK = [80, 81, 82]

        # Identify all condition events
        cond_events = events[np.isin(events[:, 2], COND)]

        # Helper: get all events inside a trial window
        def get_events_between(events, start, end):
            return events[(events[:, 0] > start) & (events[:, 0] < end)]

        # Lists to collect valid/invalid trials
        valid_trials = []
        invalid_trials = []

        # Validate trial-by-trial
        for i, cond in enumerate(cond_events):

            cond_sample = cond[0]

            # Boundary of the trial = next condition OR end of data
            if i < len(cond_events) - 1:
                next_sample = cond_events[i + 1][0]
            else:
                next_sample = np.inf

            # All events inside this trial
            trial_events = get_events_between(events, cond_sample, next_sample)

            # Required event checks
            has_cue = np.any(trial_events[:, 2] == CUE)
            has_target = np.any(trial_events[:, 2] == TARGET)
            has_feedback = np.any(np.isin(trial_events[:, 2], FEEDBACK))

            if has_cue and has_target and has_feedback:
                valid_trials.append({
                    "cond": cond,
                    "events": trial_events,
                })
            else:
                invalid_trials.append({
                    "cond": cond,
                    "events": trial_events,
                    "has_cue": has_cue,
                    "has_target": has_target,
                    "has_feedback": has_feedback
                })

        # Build the filtered event list
        filtered_events = []

        for trial in valid_trials:
            filtered_events.append(trial["cond"])       # include condition
            for ev in trial["events"]:                  # include all trial events
                filtered_events.append(ev)

        filtered_events = np.array(filtered_events)
        events = filtered_events[np.argsort(filtered_events[:, 0])]
        
    elif sub_i == 8:
        print("Sub 8")
        start = 3135
        stop = 3180

        # Cut into two halves: before and after
        raw_before = raw.copy().crop(tmin=0, tmax=start, include_tmax=False)
        raw_after = raw.copy().crop(tmin=stop, tmax=None)

        raw = mne.concatenate_raws([raw_before, raw_after])
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

    # baseline extraction
    epochs_baseline = mne.Epochs(
        raw, events, event_id=event_dict_baseline, tmin=baseline_boundary[0], tmax=baseline_boundary[1],
        baseline=None, reject_by_annotation=True, preload=True, reject=None
    )
    eeg_baseline = epochs_baseline.crop(tmin=baseline_boundary[0], tmax=baseline_boundary[1]).pick("eeg").get_data()
    eeg_baseline_mean = eeg_baseline.mean(axis=2, keepdims=True)
    condition_labels = epochs_baseline.events[:, 2]
    
    # adjust the event timing of the 50s (epoch type)
    event_codes_to_adjust_timing = [50, 51, 52, 53]
    events_adjusted = anchor_preceding_multiple(events, event_codes_to_adjust_timing)
    # extract cue and target epochs
    tmin, tmax = epoch_boundary_cue
    desired_n = 1179  # TODO: hard-coded
    tmax_fixed = tmin + (desired_n - 1) / raw.info["sfreq"]
    epochs_cue = mne.Epochs(
        raw, events_adjusted, event_id=event_dict_cue, tmin=tmin, tmax=tmax_fixed,
        baseline=None, reject_by_annotation=True, preload=True, reject=None, picks="eeg"
    )
    epochs_target = mne.Epochs(
        raw, events_adjusted, event_id=event_dict_target, tmin=epoch_boundary_target[0], tmax=epoch_boundary_target[1],
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
    parser.add_argument("-c", "--config_id", type=str, default="preprocessing-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        run(config, sub_i)