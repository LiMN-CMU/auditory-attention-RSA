import argparse
import json
from pathlib import Path
import multiprocessing
import cwt
import rsa
import rsamultiple

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=str, default="analysis-001", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
with open(config_path, "r") as f:
    config = json.load(f)

print(config)
def process_subject(sub_id):
    # print("Extracting spectral map...")
    # cwt.run(config, sub_id)

    print("Running RSA...")
    rsa.run(config, sub_id)

    print(f"=== Subject {sub_id} processing completed ===\n")

if __name__ == "__main__":
    # num_workers = min(len(config["subjects"]), multiprocessing.cpu_count())  # Use available CPUs
    # num_workers = 3
    # print(f"Using {num_workers} parallel workers.")
    
    # Create a pool of workers
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     pool.map(process_subject, config["subjects"])  # Ensuring controlled memory load

    for sub_i in config["subjects"]:
        process_subject(sub_i)

    print("=== All Preprocessing Completed ===")