# Embedding Model 1

- ResNet34StatsPool
- No light tone
- No initials
- 60K train samples
- batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=4, factor=0.1)

|||||
|---|---|---|---|
| 2986 |  342 |   37 |  271 |
|  338 | 3262 |  282 |  138 |
|   85 |  378 | 1870 |  512 |
|  379 |  190 |  392 | 4823 |

Epoch 23 Loss 0.4081 Accuracy 0.843 lr 0.010000 acc_val 0.795

# Contextual Model 1

- Transformer Encoder
- Light tone + initials
- No duration, no one-hot encoding
- 15k train utterances
- batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=4, factor=0.1)
- Epoch 246

|||||||
|---|---|---|---|---|---|
| 33124 |  109 |  178 |  132 |  156 |   8 |
|   136 | 5986 |  483 |   82 |  550 |  13 |
|   305 |  631 | 6160 |  560 |  272 |  75 |
|   166 |   95 |  623 | 3736 |  764 |  72 |
|   310 |  710 |  259 |  660 | 9388 |  68 |
|    54 |  129 |  385 |  695 |  502 | 341 |

| | precision |  recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.97 | 0.98 | 0.98 | 33707 |
| 1 | 0.78 | 0.83 | 0.80 |  7250 |
| 2 | 0.76 | 0.77 | 0.77 |  8003 |
| 3 | 0.64 | 0.68 | 0.66 |  5456 |
| 4 | 0.81 | 0.82 | 0.82 | 11395 |
| 5 | 0.59 | 0.16 | 0.25 |  2106 |
|accuracy     |      |      | 0.86 | 67917 |
|macro avg    | 0.76 | 0.71 | 0.71 | 67917 |
|weighted avg | 0.86 | 0.86 | 0.86 | 67917 |