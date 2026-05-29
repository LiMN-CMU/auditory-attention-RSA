import argparse
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description

# Parameters
def run(config, sub_i):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_bids = config["raw_to_bids"]
    in_dir = base_dir / config_bids["input_folder"]
    out_dir = base_dir / config_bids["output_folder"]

    # montage_p = base_dir / "etc"
    # with open(montage_p / "channel_dict_ABC.json", "r") as f:
    #     channel_mapping = json.load(f)
    # montage = mne.channels.read_custom_montage(montage_p / "chanlocs_64_3_eye_chan.locs")

    # Code
    make_dataset_description(
        path=out_dir,
        name=config_bids["dataset_description"]["name"],
        authors=config_bids["dataset_description"]["authors"],
        dataset_type="raw"
    )

    print(f"\n=== Processing subject: {sub_i:03} ===")
    bdf_fpaths = sorted(in_dir.glob(f"RSAS_PLT{sub_i}_*.bdf"))  # NOTE: hard-coded name format
    for run_i, bdf_fpath in enumerate(bdf_fpaths):
        raw = mne.io.read_raw_bdf(bdf_fpath, preload=False)
        
        bids_path = BIDSPath(
            subject=f"{sub_i:03}", 
            task=task, 
            datatype="eeg",
            suffix="eeg",
            root=out_dir,
            run=run_i  # NOTE: We used run argument to save sharded bdf files.
        )

        # NOTE: No changes (renamed channel, montage) are applied in the bdf data. Only the metadata are changed.
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=True,
            # allow_preload=True,
            # format='EEGLAB'
        )

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
