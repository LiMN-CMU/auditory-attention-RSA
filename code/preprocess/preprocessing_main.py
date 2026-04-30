import argparse
import json
from pathlib import Path
import raw_to_bids
import filter
import reject_channel
import calculate_ICA
import apply_ICA
import epoch

def process_subject(sub_id):
    print("Converting to BIDS format...")
    raw_to_bids.run(config, sub_id)

    print("Running filtering...")
    filter.run(config, sub_id)

    print("Rejecting channels...")
    reject_channel.run(config, sub_id, config["mode"])

    print("Calculating ICA...")
    calculate_ICA.run(config, sub_id)

    print("Applying ICA...")
    apply_ICA.run(config, sub_id, config["mode"])

    print("Epoching data...")
    epoch.run(config, sub_id, config["mode"])

    print(f"=== Subject {sub_id} processing completed ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="preprocessing-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        process_subject(sub_i)

    print("=== All Preprocessing Completed ===")
