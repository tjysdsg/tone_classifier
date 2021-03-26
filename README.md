- Place your own `phone_ctm.txt` file in project root dir, or use the default one generated
  from https://github.com/tjysdsg/aidatatang_force_align on AISHELL-3 data
- Run

```bash
  python feature_extration.py
```

to collect required statistics (phone start time, duration, tones, etc). Results are saved to `utt2tones.json`

- Run

```bash
  python trian/embedding/split_wavs.py
```

to split train, test, and validation dataset for embedding model training

- Run

```bash
python train/train_embedding.py
```

to train embedding model, the results are in `exp/embedding*/`

Mel-spectrogram cache is generated at `exp/cache/spectro/wav.scp` and `exp/cache/spectro/*.npy`

- Run `train/embedding/extract_embedding.py` to generate embeddings of utterances according to `utt2tones.json`. Results
  are in `embeddings/`
- Run

```bash
python train/train_transformer.py
```

to train the transformer model, the results are in `exp/transformer*/`

TODO: explain training script arguments