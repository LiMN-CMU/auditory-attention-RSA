import argparse
from joblib import Parallel, delayed
import json
from pathlib import Path
import mne
from mne_bids import BIDSPath
import numpy as np
from scipy.signal import filtfilt, firwin

def filter_events(events):
    # Filter out consecutive events with the same masked type
    masked_types = events[:, 2]
    change_indices = np.insert(np.diff(masked_types) != 0, 0, True)
    filtered_events = events[change_indices]

    return filtered_events

# Parameters
def run(config, sub_i):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_filter = config["filter"]
    in_dir = base_dir / config_filter["input_folder"]
    out_dir = base_dir / config_filter["output_folder"]
    
    chan_ref = config_filter["reference_channels"]
    fs_down = config_filter["target_sample_rate"]
    fc_bpf = [config_filter["frequency_low_cutoff"], config_filter["frequency_high_cutoff"]]
    filter_phase = config_filter["filter_phase"]
    filter_window = config_filter["filter_window"]  # NOTE: I used hamming window instead of kaiser (as MNE doesn"t support kaiser window).
    # filter_order = 1856  # Filter length
    # filter_kaiser_beta = 5.65326
    # NOTE: I did not specify the filter length as it is automatically determined, resulting in higher filter length (originally 1845, now 30k)

    montage_p = base_dir / "etc"
    with open(montage_p / "channel_dict_ABC.json", "r") as f:
        channel_mapping = json.load(f)
    montage = mne.channels.read_custom_montage(montage_p / "chanlocs_64_3_eye_chan.locs")

    # Load data and apply filtering
    sub_str = f"sub-{sub_i:03d}"
    eeg_list = []
    bids_path = in_dir / sub_str / "eeg"
    file_pattern = f"{sub_str}_task-{task}"
    all_files = [f for f in bids_path.iterdir() if file_pattern in f.name and f.suffix == ".bdf"]
    print(f"\n=== Processing subject: {sub_str} ===")

    for file_path in all_files:
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        eeg_list.append(raw)

    # Combine EEG files into one
    combined_raw = mne.concatenate_raws(eeg_list)
    
    # Remove unwanted channels  TODO: hard-coded
    combined_raw.drop_channels(["EXG6", "EXG7", "EXG8"])

    combined_raw.rename_channels(channel_mapping)
    combined_raw.set_channel_types(mapping=
        {"EXG1": "misc",  # reference channels 
        "EXG2": "misc",  # reference channels 
        "EXG3": "eog",
        "EXG4": "eog",
        "EXG5": "eog",
        "Status": "stim"}
    )
    combined_raw.set_montage(montage, on_missing="ignore")
    
    events = mne.find_events(combined_raw, mask=255, mask_type="and", shortest_event=1)
    filtered_events = filter_events(events)
    combined_raw.add_events(filtered_events, replace=True)

    # Apply referencing
    combined_raw.set_eeg_reference(ref_channels=chan_ref)

    # # Custom FIR filter to make similar filter
    # bpfilter = firwin(
    #     numtaps=filter_order + 1,
    #     cutoff=fc_bpf,
    #     fs=combined_raw.info['sfreq'],
    #     pass_zero=False,
    #     window=(filter_window, filter_kaiser_beta)  # Matches EEGLAB Kaiser filter
    # )

    # Apply filtfilt to each channel in parallel
    # n_jobs = -1  # Use all available CPU cores
    # eeg_data = combined_raw.get_data()
    # filtered_data_parallel = Parallel(n_jobs=n_jobs)(
    #     delayed(filtfilt)(bpfilter, [1.0], eeg_data[ch, :]) for ch in range(eeg_data.shape[0])
    # )
    # # filtered_data = filtfilt(bpfilter, [1.0], combined_raw.get_data(), axis=1)
    # filtered_data = np.array(filtered_data_parallel)
    # combined_raw._data = filtered_data  # Replace the raw data with filtered data

    # # Band-pass filter
    print('Filtering the data...')
    combined_raw.filter(
        l_freq=fc_bpf[0],
        h_freq=fc_bpf[1],
        fir_design="firwin",
        fir_window=filter_window,  
        phase=filter_phase,
        verbose=True  
    )

    # Downsample
    combined_raw.resample(fs_down)

    # Define the BIDS path for derivatives
    bids_out = BIDSPath(
        subject=sub_str.split("-")[1],
        task=task,
        datatype="eeg",
        suffix="eeg",
        extension=".fif",
        root=out_dir,
        processing=out_dir.name,
    )
    bids_out.fpath.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data in FIF format within BIDS derivatives
    combined_raw.save(bids_out.fpath.with_suffix(".fif"), overwrite=True)
    

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