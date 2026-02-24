## Champollion-Decoder

```bash
git clone https://github.com/Dufouranto0/2025_Champollion_Decoder.git
cd 2025_Champollion_Decoder
```

## Run the pixi env

```bash
cd pixi
pixi shell
cd ..
```

### Previous env

```bash
python -m venv decodervenv
. decodervenv/bin/activate
pip install -r requirements.txt
```

## Training a decoder for a specific model

To train a decoder for a specific model, modify the `model_to_decode_path` field in the config file to point to the **directory of the model** you want to decode.  

Also make sure that by concatenating `model_to_decode_path` and `train_csv`, you get the path to the **embeddings corresponding to the subjects used during the training of the encoder**.  

The `val_test_csv` corresponds to embeddings containing subjects used for **validation** in the encoder training.

Then run (inside the folder `2025_Champollion_Decoder`):
```bash
python3 -m decoder.train
```
---

## Decoding subjects from a .csv file

If you want to decode subjects, with a known model (already trained), use `decode_subjects.py`.
`decode_subjects.py` takes as argument the path the the trained model folder (-p), the list of subjects (-s), the path to the embeddings file (-e) and the name of the ID column in the embeddings file (-c):

Run (inside the folder `2025_Champollion_Decoder`):
```bash
python3 -m decoder.reconstruction.decode_subjects \
                       -p runs/Champollion_V1_after_ablation_256/23_SC-sylv_left_bce_0.0005 \
                       -s sub-1000021,sub-1000325,sub-1000575,sub-1000606 \
                       -e /neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/SC-sylv_left/name09-39-51_74/ukb40_random_embeddings/full_embeddings.csv \
                       -c ID

```

---

## Decoding subjects from train, val and test

If you want to decode all the subjects fom the train, val and test sets, with a known model (already trained), use `decode_train_val_test.py`.
`decode_train_val_test.py` takes as argument the path the the trained model folder (-p), the split you want to work on (--split), the kind of subjects you want to save regarding the loss function (i.e. the reconstruciton error), the best reconstructed subjects or the worst (-m), the number of reconstruction to save regarding this criteria (-n), and finaly a possibility to compute the smoothed score per voxel for each subject, regarding the distribution of error in each split, so as to find outliers (--outliers):

Run (inside the folder `2025_Champollion_Decoder`):
```bash
python3 -m decoder.reconstruction.decode_train_val_test \
                          -p runs/1_STs_left_bce_0.0005 \
                          -m worst \
                          -n 10
```


---

## Comparing encoder input with decoder output

To compare the initial encoder input (in black and white in the example below) with the decoder output (in orange in the example below), you also need a BrainVisa environment. It is now included in the pixi env. Otherwise, if you are still with the previous venv, use bv bash after you have downloaded brainvisa.

`visu.py` takes the parent folder of the .npy files generated with `decode_subjects.py` or `decode_train_val_test.py`, the loss function that was used during the training and a potential list of subject ids.
If no loss is provided, then bce is used by default, if no list of subjects is provided, 4 subjects among the .npy files will be randomly picked up.

Run (inside the folder `2025_Champollion_Decoder`):
```bash
python3 decoder/reconstruction/visu.py \
  -p decoder/example \
  -s sub-1110622,sub-1150302
```

## Decoder Architecture

The architecture of the decoder (file `convnet.py`) is described in the following figure: 

![Decoder Architecture](figures/decoder_architecture.png)

## Example of visualization

In the following figure, 4 input shapes (in black and white, on the left) were encoded using the Champollion V1 architecture, and decoded using this decoder.
The decoder output after 10 epochs is provided here, not as a binary image, but as a continuous image containing the probability of each voxel to be 1 in the inital image.
The higher the probability, the redder the voxel appears. The lower the probability, the more transparent and yellow the voxel appears.

![Decoder Output Example](figures/SOr_left.png)

---

## Saving NIfTI files from decoder outputs

If you want to save NIfTI files from the NumPy outputs of the decoder, use `save_nii.py` inside a BrainVisa environment (bv bash) (or any environment that contains `aims`).
Note that it is now included in the pixi env.
`save_nii.py` takes the parent folder of the generated .npy files as argument:

```bash
cd 2025_Champollion_Decoder/decoder
python3 reconstruction/save_nii.py -p example
```

