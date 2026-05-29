import argparse
import json
from pathlib import Path
import cwt
import rsa

def process_subject(sub_id):
    print("Extracting spectral map...")
    cwt.run(config, sub_id)

    print("Running RSA...")
    rsa.run(config, sub_id)

    print(f"=== Subject {sub_id} processing completed ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="analysis-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    for sub_i in config["subjects"]:
        process_subject(sub_i)

    print("=== All Preprocessing Completed ===")