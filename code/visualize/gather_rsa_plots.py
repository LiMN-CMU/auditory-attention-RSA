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

    feat_i = config_rsa["frequency_band_index"]
    model_type = config_rsa["decoder_model"]
    categories = config_rsa["auditory_attention_categories"]
    epoch_type_dict = {'cue': config_rsa["epoch_boundary_cue"], 'target': config_rsa["epoch_boundary_target"]}
    plot_gather_type = config_rsa["plot_gather_type"]

    # Timepoints and categories
    for epoch_type, epoch_boundary in epoch_type_dict.items():
        target_times = config_rsa[f"plot_target_time_{epoch_type}"]
        
        if plot_gather_type == "vertical":
            # RDM
            fig_rdm, ax_rdm = plt.subplots(len(target_times), 1, figsize=(3.5, 12), sharex=True)
            tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
            for i, t in enumerate(target_times):
                img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm_{t:.1f}s.png')
                ax_rdm[i].imshow(img)
                ax_rdm[i].axis('off')
            fig_rdm.subplots_adjust(left=0.3)
            for pos, label in zip(tick_positions, target_times):
                fig_rdm.text(0.18, pos, label, fontsize=16, va='center', ha='right')
                
            # model weight matrix
            fig_model, ax_model = plt.subplots(len(target_times), len(categories), figsize=(18, 12))
            for row, t in enumerate(target_times):
                for col, cat in enumerate(categories):
                    fname = in_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_{t:.1f}s_category-{cat}.png'
                    img = load_image(fname)
                    ax_model[row, col].imshow(img)
                    ax_model[row, col].axis('off')
                    # if row == 0:
                    #     ax_model[row, col].set_title(cat.replace("-", "\n"), fontsize=16)

            # Add time tick labels on the left side
            fig_model.subplots_adjust(left=0.15)
            for pos, label in zip(tick_positions, target_times):
                fig_model.text(0.11, pos, label, fontsize=20, va='center', ha='right')

        elif plot_gather_type == "horizontal":
            # RDM
            fig_rdm, ax_rdm = plt.subplots(1, len(target_times), figsize=(3.5 * len(target_times), 3.5), sharey=True)
            tick_positions = np.linspace(0.82, 0.17, 6)  # from top to bottom
            for i, t in enumerate(target_times):
                img = load_image(in_dir / f'group_task-craa_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm_{t:.1f}s.png')
                ax_rdm[i].imshow(img)
                ax_rdm[i].axis('off')
                ax_rdm[i].set_title(f'{t * 1000:.0f} ms', fontsize=20)
            fig_rdm.savefig(out_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_rdm_total.png', dpi=300, transparent=True)
            
            # SVM Weights
            fig_model, ax_model = plt.subplots(len(categories), len(target_times), figsize=(3.5 * len(target_times), 3.5 * len(categories)))

            # Make sure ax_model is always a 2D array
            if len(categories) == 1:
                ax_model = np.expand_dims(ax_model, axis=0)
            if len(target_times) == 1:
                ax_model = np.expand_dims(ax_model, axis=1)

            for row, cat in enumerate(categories):
                for col, t in enumerate(target_times):
                    fname = in_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_{t:.1f}s_category-{cat}.png'
                    img = load_image(fname)
                    ax_model[row][col].imshow(img)
                    ax_model[row][col].axis('off')
                    if row == 0:
                        ax_model[row][col].set_title(f'{t * 1000:.0f} ms', fontsize=16)
                # Add category labels on the left side
                ax_model[row][0].set_ylabel(cat.replace("-", "\n"), fontsize=14, rotation=0, labelpad=40, va='center')
            fig_model.subplots_adjust(left=0.2, top=0.88)

            # fig_model.suptitle('Model Weights', fontsize=16)
            fig_model.savefig(
                out_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_total.png',
                dpi=500,
                transparent=False
                )
            plt.close()

            # masked SVM Weights
            fig_model, ax_model = plt.subplots(len(categories), len(target_times), figsize=(3.5 * len(target_times), 3.5 * len(categories)))

            # Make sure ax_model is always a 2D array
            if len(categories) == 1:
                ax_model = np.expand_dims(ax_model, axis=0)
            if len(target_times) == 1:
                ax_model = np.expand_dims(ax_model, axis=1)

            for row, cat in enumerate(categories):
                for col, t in enumerate(target_times):
                    fname = in_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_{t:.1f}s_category-{cat}-masked.png'
                    img = load_image(fname)
                    ax_model[row][col].imshow(img)
                    ax_model[row][col].axis('off')
                    if row == 0:
                        ax_model[row][col].set_title(f'{t * 1000:.0f} ms', fontsize=16)
                # Add category labels on the left side
                ax_model[row][0].set_ylabel(cat.replace("-", "\n"), fontsize=14, rotation=0, labelpad=40, va='center')
            fig_model.subplots_adjust(left=0.2, top=0.88)

            # fig_model.suptitle('Model Weights', fontsize=16)
            fig_model.savefig(
                out_dir / f'group_task-{task}_desc-{epoch_type}_feat-{feat_i}_model-{model_type}_config-{config_id}_haufeweights_total-masked.png',
                dpi=500,
                transparent=False
                )
        


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