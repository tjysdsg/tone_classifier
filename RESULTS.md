# Baseline 1

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

| | precision |  recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.79 | 0.82 | 0.80 | 7516  |
| 1 | 0.78 | 0.80 | 0.79 | 8103  |
| 2 | 0.71 | 0.66 | 0.69 | 5474  |
| 3 | 0.85 | 0.84 | 0.84 | 11548 |
|accuracy | | | 0.80 | 32641 |
| macro avg | 0.78 | 0.78 | 0.78 | 32641 |
| weighted avg | 0.79 | 0.80 | 0.79 | 32641 |

Epoch 23 Loss 0.4081 Accuracy 0.843 lr 0.010000 acc_val 0.795

# Baseline 3

- Epoch 14
- ResNet34StatsPool
- Neutral tone + initials
- Segment duration + onehot encoding
- Batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=2, factor=0.1)

>       [21706     0     0     0     0     0]
>       [    0  4085   282    53   360     7]
>       [    0   178  4402   227   276    42]
>       [    0    72   319  2470   596    20]
>       [    0   225   174   197  6682    17]
>       [    0    19    59    45    94  1190]
>
>                  precision    recall  f1-score   support
>       0            1.00      1.00      1.00     21706
>       1            0.89      0.85      0.87      4787
>       2            0.84      0.86      0.85      5125
>       3            0.83      0.71      0.76      3477
>       4            0.83      0.92      0.87      7295
>       5            0.93      0.85      0.89      1407
>
>       accuracy                         0.93     43797
>       macro avg    0.89      0.86      0.87     43797
>       weighted avg 0.93      0.93      0.92     43797

Epoch 14 Loss 0.1594 Accuracy 0.945 lr 0.010000 acc_val 0.926

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