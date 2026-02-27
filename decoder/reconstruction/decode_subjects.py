# decode_subjects.py

"""
Example:
cd 2025_Champollion_Decoder

python3 -m decoder.reconstruction.decode_subjects \
                -p runs/Champollion_V1_after_ablation_256/57_fronto-parietal_medial_face_left_bce_0.0005 \
                -s sub-1000021,sub-1000325,sub-1000575,sub-1000606,sub-1000715,sub-1000963 \
                -e /neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/fronto-parietal_medial_face_left/name16-13-44_35/ukb40_random_embeddings/train_embeddings.csv \
                -c ID
"""

import argparse
import os
import yaml
import torch
import pandas as pd
import numpy as np

from decoder.model.convnet import Decoder


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_paths(run_dir):
    best_model_path = os.path.join(run_dir, "best_model.pth")
    decoder_config_path = os.path.join(run_dir, ".hydra", "decoder_config.yaml")
    encoder_config_path = os.path.join(run_dir, ".hydra", "encoder_config.yaml")
    recon_dir = os.path.join(run_dir, "reconstruction_best_model")
    os.makedirs(recon_dir, exist_ok=True)

    return best_model_path, decoder_config_path, encoder_config_path, recon_dir


def load_embeddings(embeddings_csv, subject_column, selected_subjects=None):
    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(embeddings_csv)

    print(f"Using embeddings from: {embeddings_csv}")
    df = pd.read_csv(embeddings_csv)

    if subject_column not in df.columns:
        raise ValueError(f"The column {subject_column} is not found in {embeddings_csv}")

    df[subject_column] = df[subject_column].astype(str)
    if selected_subjects:
        subject_list = selected_subjects.split(',')
        df = df[df[subject_column].isin(subject_list)]

    if df.empty:
        raise ValueError("No embeddings found for given subjects!")

    return df


def compute_output_shape(encoder_cfg, decoder_cfg):
    region = list(encoder_cfg["dataset"].keys())[0]
    s = encoder_cfg["dataset"][region]["input_size"]

    output_shape = tuple(map(int, s.strip("()").split(",")))
    output_shape = output_shape[1:][::-1]

    if decoder_cfg["loss"] in ["bce", "mse"]:
        output_shape = (1,) + output_shape
    elif decoder_cfg["loss"] == "ce":
        output_shape = (2,) + output_shape

    return output_shape


def load_model(best_model_path, encoder_cfg, decoder_cfg, output_shape):
    model = Decoder(
        latent_dim=encoder_cfg["backbone_output_size"],
        output_shape=output_shape,
        filters=encoder_cfg["filters"][::-1],
        drop_rate=decoder_cfg["dropout"],
        loss_name=decoder_cfg["loss"],
    )

    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()

    return model


def decode_and_save(model, df, subject_column, decoder_cfg, recon_dir, batch_size=32):

    embeddings = torch.tensor(
        df.drop(columns=[subject_column]).values,
        dtype=torch.float32
    )

    subjects = df[subject_column].tolist()
    n_samples = len(embeddings)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_subjects = subjects[start:end]
        batch_embeddings = embeddings[start:end]

        with torch.no_grad():
            outputs = model(batch_embeddings).cpu()

        for subj_id, out in zip(batch_subjects, outputs):

            if decoder_cfg["loss"] == "mse":
                output_vol = out.numpy()[0].astype(np.float32)

            elif decoder_cfg["loss"] == "bce":
                raw = torch.sigmoid(out)
                output_vol = raw.numpy()[0].astype(np.float32)

            elif decoder_cfg["loss"] == "ce":
                output_vol = out[1].numpy().astype(np.float32)

            else:
                raise ValueError(f"Unsupported loss function: {decoder_cfg['loss']}")

            # (D, H, W) -> (Z, Y, X)
            output_vol = output_vol.transpose(2, 1, 0)

            out_path = os.path.join(recon_dir, f"{subj_id}_decoded.npy")
            np.save(out_path, output_vol)
            print(f"Saved {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Decode subjects with best model")
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-s", "--subjects", type=str, default=None)
    parser.add_argument("-e", "--embeddings", required=True)
    parser.add_argument("-c", "--IDcolumnName", default="Subject")

    args = parser.parse_args()

    # 1️ Prepare paths
    best_model_path, decoder_config_path, encoder_config_path, recon_dir = \
        prepare_paths(args.path)

    # 2️ Load configs
    encoder_cfg = load_config(encoder_config_path)
    decoder_cfg = load_config(decoder_config_path)

    # 3️ Load embeddings
    df = load_embeddings(
        args.embeddings,
        args.IDcolumnName,
        args.subjects
    )

    # 4️ Compute output shape
    output_shape = compute_output_shape(encoder_cfg, decoder_cfg)

    # 5️ Load model
    model = load_model(
        best_model_path,
        encoder_cfg,
        decoder_cfg,
        output_shape
    )

    # 6️ Decode and save
    decode_and_save(
        model,
        df,
        args.IDcolumnName,
        decoder_cfg,
        recon_dir
    )


if __name__ == "__main__":
    main()