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
    categories = config_rsa["auditory_attention_categories"]
    epoch_types = ["cue", "target"]

    # Timepoints and categories
    for epoch_type in epoch_types:
        target_times = config_rsa[f"target_time_{epoch_type}"]
        for feat_i in feat_indices:
            # # 1) vertical
            # fig_rdm, ax_rdm = plt.subplots(len(target_times), 1, figsize=(3.5, 12), sharex=True)
            # tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
            # for i, t in enumerate(target_times):
            #     img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_conf-{config_id}_rdm_{t:.1f}s.png')
            #     ax_rdm[i].imshow(img)
            #     ax_rdm[i].axis('off')
            # fig_rdm.subplots_adjust(left=0.3)
            # for pos, label in zip(tick_positions, target_times):
            #     fig_rdm.text(0.18, pos, label, fontsize=16, va='center', ha='right')

            # 2) horizontal 
            fig_rdm, ax_rdm = plt.subplots(1, len(target_times), figsize=(3.5 * len(target_times), 3.5), sharey=True)
            tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
            if len(target_times) == 1:
                ax_rdm = [ax_rdm]  # Make iterable if only one subplot
            for i, t in enumerate(target_times):
                img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_features-{feat_indices}_feat-{feat_i}-only_model-{model_type}_conf-{config_id}_rdm_{t:.1f}s.png')
                ax_rdm[i].imshow(img)
                ax_rdm[i].axis('off')
                # ax_rdm[i].set_title(f'{t * 1000:.0f} ms', fontsize=14)

            # fig_rdm.subplots_adjust(top=0.85)

            # Add large "Time" label
            # if feat_i == 2:
            #     fig_rdm.suptitle('Alpha (8 - 14 Hz)', fontsize=16)
            # elif feat_i == 0:
            #     fig_rdm.suptitle('Gamma (20 - 50 Hz)', fontsize=16) 
            # elif feat_i == 1:
            #     fig_rdm.suptitle('Beta (14 - 20 Hz)', fontsize=16)    
            # elif feat_i == 3:
            #     fig_rdm.suptitle('Theta (4 - 8 Hz)', fontsize=16)            
            # elif feat_i == 4:
            #     fig_rdm.suptitle('Delta (2 - 4 Hz)', fontsize=16)
            # elif feat_i == -1:
            #     fig_rdm.suptitle('EEG Time Course', fontsize=16)
            fig_rdm.savefig(out_dir / f'group_task-{task}_desc-{epoch_type}_features-{feat_indices}_feat-{feat_i}_model-{model_type}_conf-{config_id}_rdm_total.png', dpi=300)


if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="visualize-001", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id
    
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_id}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    run(config, config["subjects"])