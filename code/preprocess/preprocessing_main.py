import argparse
import json
from pathlib import Path
import raw_to_bids
import filter
import reject_channel
import calculate_ICA
import apply_ICA
import epoch
import multiprocessing

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=str, default="preprocessing-001", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
with open(config_path, "r") as f:
    config = json.load(f)

def process_subject(sub_id):
    # print("Converting to BIDS format...")
    raw_to_bids.run(config, sub_id)

    # print("Running filtering...")
    filter.run(config, sub_id)

    # print("Rejecting channels...")
    reject_channel.run(config, sub_id, config["mode"])

    # print("Calculating ICA...")
    calculate_ICA.run(config, sub_id)

    # print("Applying ICA...")
    apply_ICA.run(config, sub_id, config["mode"])

    print("Epoching data...")
    epoch.run(config, sub_id, config["mode"])

    print(f"=== Subject {sub_id} processing completed ===\n")


if __name__ == "__main__":
    # num_workers = min(len(config["subjects"]), multiprocessing.cpu_count())  # Use available CPUs
    # num_workers = 1
    # print(f"Using {num_workers} parallel workers.")
    
    # Create a pool of workers
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     pool.map(process_subject, config["subjects"])
    for sub_i in config["subjects"]:
        process_subject(sub_i)

    print("=== All Preprocessing Completed ===")
