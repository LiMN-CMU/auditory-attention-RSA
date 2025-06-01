import argparse
import json
from pathlib import Path
import mne
from mne_bids import BIDSPath
from mne.preprocessing import ICA
from mne import set_config

set_config('MNE_USE_NUMBA', 'true')  # Enable JIT acceleration
# set_config('MNE_NUM_THREADS', '4')   # Use 4 CPU threads TODO: hard-coded

# Parameters
def run(config, sub_i):
    task = config["task"]
    subjects = config["subjects"]

    base_dir = Path(config["base_dir"])
    config_ICA = config["calculate_ICA"]
    in_dir = base_dir / config_ICA["input_folder"]
    out_dir = base_dir / config_ICA["output_folder"]
    
    fc_bpf = [config_ICA["additional_frequency_low_cutoff"], config_ICA["additional_frequency_high_cutoff"]]
    random_state = config_ICA["ICA_random_state"]
    ica_method = config_ICA["ICA_method"]
    max_iter = config_ICA["ICA_max_iteration"]
    stop_tol = config_ICA["ICA_stop_tolerance"]

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
        processing=out_dir.name,
        extension='.fif',
        suffix='eeg',
        datatype='eeg',
        root=out_dir
    )
    out_file = bids_out.fpath
    out_file.parent.mkdir(parents=True, exist_ok=True) 

    print(f"Loading file: {in_file}")
    raw = mne.io.read_raw_fif(in_file, preload=True)
    # calculate ICA on filtered data
    filt_raw = raw.copy().filter(l_freq=fc_bpf[0], h_freq=fc_bpf[1])

    # Run ICA and ICLabel
    n_eeg_chan = len(mne.pick_types(raw.info, eeg=True))
    ica = ICA(
        n_components=n_eeg_chan, 
        method=ica_method, 
        fit_params=dict(extended=True, w_change=stop_tol),  # extended infomax
        max_iter=max_iter,
        random_state=random_state,
        verbose=True
    )
    ica.fit(filt_raw)

    # Save
    ica_fname = out_file.parent / f"{out_file.stem}-ica.fif"  # ICA solution
    ica.save(ica_fname, overwrite=True)
    raw.save(out_file.with_suffix(".fif"), overwrite=True)

    print(f"Processed and saved: {out_file}")


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