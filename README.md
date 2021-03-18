- Run `show_phone_alignment.sh` to generate `phone_ctm.txt`
- Run

```bash
  python feature_extration.py --stage 1
```

to collect required statistics

- Run

```bash
  python feature_extration.py --stage 2
```

to perform feature extraction, the results are in `feats/`

- Run `fix_data_dir.py` to generate `wav.scp`, `utt2spk`, and `spk2utt` files under `feats/`
- Run

```bash
python train/train_embedding.py
```

to train embedding model, the results are in `exp/embedding/`

- Run `fix_utt2tones.py` to generate `utt2tones_fixed.json` that contains only the generated `.npy` files under `feats/`
- Run

```bash
python train/train_transformer.py
```

to train the final model, the results are in `exp/transformer/`
