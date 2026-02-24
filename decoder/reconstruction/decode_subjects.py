# decode_subjects.py

"""
Problem with the subject ID column type. It is int for HCP subjects and string for UKB subjects.
To FIX !
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


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Decode subjects with best model")
    parser.add_argument("-p", "--path", required=True,
                        help="Path to run folder (e.g. runs/57_fronto-parietal_medial_face_left_bce_0.0005)")
    parser.add_argument("-s", "--subjects",  nargs="+", default=None, type=str,
                        help="List of subject IDs to decode, separated with a ,")
    parser.add_argument("-e", "--embeddings", required=True,
                        help="Path to the embeddings file")
    parser.add_argument("-c", "--IDcolumnName", default='Subject',
                        help="Name of the subject ID column in the embeddings file")
    args = parser.parse_args()

    run_dir = args.path
    best_model_path = os.path.join(run_dir, "best_model.pth")
    decoder_config_path = os.path.join(run_dir, ".hydra", "decoder_config.yaml")
    encoder_config_path = os.path.join(run_dir, ".hydra", "encoder_config.yaml")
    recon_dir = os.path.join(run_dir, "reconstruction_best_model")
    os.makedirs(recon_dir, exist_ok=True)

    # --- Load config
    encoder_cfg = load_config(encoder_config_path)
    decoder_cfg = load_config(decoder_config_path)

    embeddings_csv = args.embeddings
    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(embeddings_csv)
    print(f"Using embeddings from: {embeddings_csv}")

    # --------- Load embeddings ---------
    df = pd.read_csv(embeddings_csv)

    subj_ID = args.IDcolumnName
    if subj_ID not in list(df.columns):
        raise ValueError(f"The column {subj_ID} is not found in {embeddings_csv}")
    if args.subjects:
        df = df[df[subj_ID].isin(args.subjects[0].split(','))] # .isin(args.subjects.split(','))] #.isin(args.subjects)
    if df.empty:
        raise ValueError("No embeddings found for given subjects!")

    region = list(encoder_cfg["dataset"].keys())[0]
    s = encoder_cfg["dataset"][region]["input_size"]
    output_shape = tuple(map(int, s.strip("()").split(",")))
    output_shape = output_shape[1:][::-1]

    if decoder_cfg["loss"] in ["bce", "mse"]:
        output_shape = (1,) + output_shape

    if decoder_cfg["loss"] == "ce":
        # target is (D, H, W) → output should be (C=2, D, H, W)
        output_shape = (2,) + output_shape

    # --------- Load model ---------
    model = Decoder(latent_dim=encoder_cfg["backbone_output_size"],
                 output_shape=output_shape,
                 filters= encoder_cfg["filters"][::-1],
                 drop_rate=decoder_cfg["dropout"],
                 loss_name=decoder_cfg["loss"]) 
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()

    embeddings = torch.tensor(df.drop(columns=[subj_ID]).values, dtype=torch.float32)
    subjects = df[subj_ID].tolist()

    batch_size = 32
    n_samples = len(embeddings)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_subjects = subjects[start:end]
        batch_embeddings = embeddings[start:end]

        # --- Decode
        with torch.no_grad():
            outputs = model(batch_embeddings).cpu()

        # --- Save reconstructions
        for subj_id, out in zip(batch_subjects, outputs):

            if decoder_cfg["loss"] == "mse":
                values = out.cpu().numpy()
                output_vol = values[0].astype(np.float32)

            elif decoder_cfg["loss"] == "bce":
                raw = torch.sigmoid(out)
                values = raw.cpu().numpy()
                output_vol = values[0].astype(np.float32)

            elif decoder_cfg["loss"] == "ce":
                pred = out[1, :, :, :].cpu().numpy()
                output_vol = pred.astype(np.float32)

            else:
                raise ValueError(f"Unsupported loss function: {decoder_cfg['loss']}")

            # Reorder axes: (D, H, W) --> (Z, Y, X)
            output_vol = output_vol.transpose(2, 1, 0)

            # Save npy
            out_path = os.path.join(recon_dir, f"{subj_id}_decoded.npy")
            np.save(out_path, output_vol)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()