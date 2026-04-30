import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import argparse
from pathlib import Path

def load_image(filename):
    return np.array(Image.open(filename))

def run(config, sub_inds):
    # Extract config values
    task = config["task"]
    base_dir = Path(config["base_dir"])
    config_id = config["configuration_id"]
    config_rsa = config["visualize_rsa"]
    
    in_dir = base_dir / config_rsa["output_folder"]
    out_dir = base_dir / config_rsa["output_folder"] / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_indices = config_rsa["frequency_band_indices"]
    model_type = config_rsa["decoder_model"]
    categories = config_rsa["categories_of_interest"]
    epoch_types = ["cue", "target"]
    plot_gather_type = config_rsa["plot_gather_type"]

    # Timepoints and categories
    for epoch_type in epoch_types:
        target_times = config_rsa[f"plot_target_time_{epoch_type}"]
        for feat_i in feat_indices:
            if plot_gather_type == "vertical":
                fig_rdm, ax_rdm = plt.subplots(len(target_times), 1, figsize=(3.5, 12), sharex=True)
                tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
                for i, t in enumerate(target_times):
                    img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_features-{feat_indices}_feat-{feat_i}-only_model-{model_type}_config-{config_id}_rdm_{t:.1f}s.png')
                    ax_rdm[i].imshow(img)
                    ax_rdm[i].axis('off')
                fig_rdm.subplots_adjust(left=0.3)
                for pos, label in zip(tick_positions, target_times):
                    fig_rdm.text(0.18, pos, label, fontsize=16, va='center', ha='right')

            elif plot_gather_type == "horizontal":
                fig_rdm, ax_rdm = plt.subplots(1, len(target_times), figsize=(3.5 * len(target_times), 3.5), sharey=True)
                tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
                for i, t in enumerate(target_times):
                    img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_features-{feat_indices}_feat-{feat_i}-only_model-{model_type}_config-{config_id}_rdm_{t:.1f}s.png')
                    ax_rdm[i].imshow(img)
                    ax_rdm[i].axis('off')
                    # ax_rdm[i].set_title(f'{t * 1000:.0f} ms', fontsize=14)

                fig_rdm.savefig(out_dir / f'group_task-{task}_desc-{epoch_type}_features-{feat_indices}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm_total.png', dpi=300, transparent=True)
                print(f'Figure saved in {out_dir}')

if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_id", type=str, default="visualize-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id
    
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    run(config, config["subjects"])