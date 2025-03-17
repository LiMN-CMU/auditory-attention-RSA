import json
from pathlib import Path
import raw_to_bids
import filter
import reject_channel
import calculate_ICA
import apply_ICA
import epoch

# Load config
config_path = Path("../../config/preprocessing-001.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Run preprocessing steps
print("=== Changing to BIDS format ===")
raw_to_bids.run(config)

print("=== Running Preprocessing ===")
filter.run(config)
reject_channel.run(config, config["mode"])
calculate_ICA.run(config)
apply_ICA.run(config, config["mode"])
epoch.run(config, config["mode"])

print("=== Preprocessing Completed ===")