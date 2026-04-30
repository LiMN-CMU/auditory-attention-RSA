import argparse
from pathlib import Path
import mne
import numpy as np
import json
from mne_bids import BIDSPath

# Parameters
def run(config, sub_i, mode="manual"):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_ch = config["reject_channel"]
    in_dir = base_dir / config_ch["input_folder"]
    out_dir = base_dir / config_ch["output_folder"]    
    
    var_sd_thres = config_ch["variance_threshold"]
    amp_thres = config_ch["amplitude_threshold"]  # µV

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

    print(f"Loading file: {in_file}")
    raw = mne.io.read_raw_fif(in_file, preload=True, verbose=False)
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, ecg=False, stim=False, exclude=[])
    data_eeg = raw.get_data(picks=picks_eeg)  # shape: (n_eeg_channels, n_samples)
    eeg_ch_names = [raw.ch_names[p] for p in picks_eeg]
    n_channels = len(eeg_ch_names)

    # Compute variance per channel
    variances = np.var(data_eeg, axis=1)  # shape: (n_channels,)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    upper_threshold = mean_var + var_sd_thres * std_var

    high_var = []
    for idx, var_val in enumerate(variances):
        if var_val > upper_threshold:
            ch_name = eeg_ch_names[idx]
            high_var.append(ch_name)

    # Check absolute amplitude
    bad_amp = []
    for ch_idx in range(n_channels):
        ch_name = raw.ch_names[ch_idx]
        if np.any(np.abs(data_eeg[ch_idx, :]) > amp_thres):
            bad_amp.append(ch_name)

    # Combine flagged channels (union of all criteria)
    flagged_channels = list(set(bad_amp) & set(high_var))
    print("=== Channels flagged by amplitude:", bad_amp)
    print("=== Channels flagged by high variance:", high_var)
    print("=== Overall flagged channels:", flagged_channels)
    if flagged_channels and mode == "manual":
        # Visual inspection (with bad channels being red)
        raw.info['bads'] = flagged_channels
        raw.plot(bad_color='red', block=True) # decide whether to remove the channel or not
        removed_channels = raw.info["bads"]  # final channels to be removed
    else:
        # automatic bad channel flag
        removed_channels = flagged_channels

    if removed_channels:
        print(f"Dropped channels: {removed_channels}")
        raw.info['bads'] = list(set(removed_channels))
        # raw.interpolate_bads(reset_bads=True)  # interplate bad channels
        
        reject_meta_dict = {
            "ChannelsRemoved": sorted(removed_channels),
            "ChannelsFlaggedVariance": sorted(list(high_var)),
            "ChannelsFlaggedAmplitude": sorted(list(bad_amp)),
        }
        json_outfile = out_file.with_suffix(".json")
        with open(json_outfile, "w") as f:
            json.dump(reject_meta_dict, f, indent=4)

        print(f"Saved metadata to: {json_outfile}")
    else:
        print("No channels removed.")

    # save file and metadata
    raw.save(out_file, overwrite=True)


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
