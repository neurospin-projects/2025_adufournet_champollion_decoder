# train.py

import os
from omegaconf import OmegaConf
from decoder.dataloader.dataloader import DataModule_Learning
from decoder.model.convnet import Decoder
from decoder.utils.train_utils import train_model
from decoder.reconstruction.save_npy import save_npy
import shutil
import re

print(">>> decoder/train.py has started running <<<")

def load_configs(decoder_cfg_path: str):
    """
    Load decoder config and corresponding encoder config.
    """
    decoder_cfg = OmegaConf.load(decoder_cfg_path)
    
    encoder_config_path = os.path.join(
        decoder_cfg.model_to_decode_path, ".hydra", "config.yaml"
    )
    if not os.path.exists(encoder_config_path):
        raise FileNotFoundError(
            f"Missing Hydra config file: {encoder_config_path}"
        )

    encoder_cfg = OmegaConf.load(encoder_config_path)
    encoder_cfg["dataset_folder"] = decoder_cfg["dataset_folder"]

    return decoder_cfg, encoder_cfg

def save_configs(out_dir, decoder_cfg, encoder_cfg):
    """Save decoder and encoder configs in Hydra-style directory."""
    hydra_dir = os.path.join(out_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)

    # Save the main config
    OmegaConf.save(config=decoder_cfg, f=os.path.join(hydra_dir, "decoder_config.yaml"))

    # Optionally also save the encoder config for traceability
    OmegaConf.save(config=encoder_cfg, f=os.path.join(hydra_dir, "encoder_config.yaml"))

    # Save overrides if needed (e.g., command line args)
    #with open(os.path.join(hydra_dir, "overrides.yaml"), "w") as f:
    #    f.write("# Add CLI overrides here if used\n")

    # Save a copy of the config.yaml used to launch the job (for convenience)
    #shutil.copy("configs/config.yaml", os.path.join(hydra_dir, "original_config.yaml"))

def infer_shapes(dm: DataModule_Learning, decoder_cfg, encoder_cfg):
    """
    Infer latent dimension, output shape, and filters from dataloader + configs.
    """
    train_loader = dm.train_dataloader()

    region = list(encoder_cfg.dataset.keys())[0]
    filters = encoder_cfg["filters"][::-1]

    latent_dim = train_loader.dataset[0][0].shape[0]
    full_target = train_loader.dataset[0][1]

    if decoder_cfg["loss"] == "ce":
        # target is (D, H, W) --> output should be (C=2, D, H, W)
        output_shape = (2,) + full_target.shape
    else:
        output_shape = full_target.shape

    return latent_dim, output_shape, filters, region


def get_next_exp_number(region_dir):
    os.makedirs(region_dir, exist_ok=True)

    region_name = os.path.basename(region_dir)

    exp_folders = [
        f for f in os.listdir(region_dir)
        if os.path.isdir(os.path.join(region_dir, f))
    ]

    numbers = []
    for folder in exp_folders:
        # match region_XX where XX is a number at the end
        match = re.match(rf"^{re.escape(region_name)}_(\d+)$", folder)
        if match:
            numbers.append(int(match.group(1)))

    if numbers:
        return max(numbers) + 1
    else:
        return 1


def main():
    # Get the absolute path to the current file
    CURRENT_FILE = os.path.abspath(__file__)
    # Go two levels up: decoder/train.py → decoder → 2025_Champollion_Decoder
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
    # --- Load configs ---
    decoder_cfg, encoder_cfg = load_configs(f"{PROJECT_ROOT}/decoder/configs/config.yaml")
    region = list(encoder_cfg.dataset.keys())[0]

    print("Region:", region)

    dataset_info = OmegaConf.to_container(
        encoder_cfg.dataset[region], resolve=True
    )

    # --- Setup data ---
    dm = DataModule_Learning(decoder_cfg, dataset_info)
    dm.setup()

    latent_dim, output_shape, filters, region = infer_shapes(
        dm, decoder_cfg, encoder_cfg
    )

    print("latent_dim:", latent_dim)
    print("output_shape:", output_shape)

    loss = decoder_cfg.loss
    print("loss_name:", loss) 


    # --- Init model ---
    model = Decoder(
        latent_dim=latent_dim,
        output_shape=output_shape,
        filters=filters,
        loss_name=loss,
        drop_rate=decoder_cfg.dropout,
    )

    # --- Training ---
    base_out_dir = os.path.join(PROJECT_ROOT, decoder_cfg.out_dir)

    # Example: PROJECT_ROOT + "runs" → "2025_Champollion_Decoder/runs"
    os.makedirs(base_out_dir, exist_ok=True)

    nb_exp = get_next_exp_number(os.path.join(base_out_dir, region))
    experiment_name = f"{region}_{nb_exp}"
    out_dir = os.path.join(base_out_dir, experiment_name)
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"Logging to: {out_dir}")

    # --- Save configs ---
    save_configs(out_dir, decoder_cfg, encoder_cfg)

    train_model(
        model,
        dm.train_dataloader(),
        dm.val_dataloader(),
        num_epochs=decoder_cfg.num_epochs,
        lr=decoder_cfg.learning_rate,
        loss_name=loss,
        out_dir=out_dir,
        save_best_model=decoder_cfg.save_best_model,
    )

    # --- Reconstructions ---
    recon_dir = os.path.join(out_dir, f"reconstructions_epoch{decoder_cfg.num_epochs}")
    save_npy(
        model,
        dm.val_dataloader(),
        device="cuda",
        out_path=recon_dir,
        save_inputs=True,
        loss_name=loss,
    )


if __name__ == "__main__":
    main()