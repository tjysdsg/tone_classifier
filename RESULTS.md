# Baseline 1

- Epoch 14
- ResNet34StatsPool
- Batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=2, factor=0.1)
- Accuracy: 8448533714805904

>     [21968   123   267   174   265    69]
>     [  127  3907   447    48   412    28]
>     [  342   374  4071   350   212   136]
>     [  220    74   439  2023   751   189]
>     [  260   416   154   438  6393   159]
>     [   90    63    90   192   249   617]

>               precision    recall  f1-score   support
>     0           0.95      0.96      0.96     22866
>     1           0.79      0.79      0.79      4969
>     2           0.74      0.74      0.74      5485
>     3           0.63      0.55      0.58      3696
>     4           0.77      0.82      0.79      7820
>     5           0.52      0.47      0.49      1301
>
>     accuracy                        0.84     46137
>     macro avg    0.73     0.72      0.73     46137
>     weighted avg 0.84     0.84      0.84     46137

# Baseline 3

- Epoch 14
- ResNet34StatsPool
- Segment duration + onehot encoding
- Batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=2, factor=0.1)
- Accuracy: 0.9260680148254112

>       [22866     0     0     0     0     0]
>       [    2  4200   324    46   391     6]
>       [    0   193  4784   190   276    42]
>       [    0    77   306  2648   646    19]
>       [    0   246   196   228  7127    23]
>       [    0    16    44    47    93  1101]

>                precision    recall  f1-score   support
>     0            1.00      1.00      1.00     22866
>     1            0.89      0.85      0.87      4969
>     2            0.85      0.87      0.86      5485
>     3            0.84      0.72      0.77      3696
>     4            0.84      0.91      0.87      7820
>     5            0.92      0.85      0.88      1301
>
>     accuracy                       0.93     46137
>     macro avg    0.89    0.87      0.88     46137
>     weighted avg 0.93    0.93      0.93     46137

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