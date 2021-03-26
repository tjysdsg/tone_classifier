# ToneNet

- accuracy: 0.8501510844283641
- precision: 0.8398256061521939
- recall: 0.837244851387725
- F1 score: 0.8385332430766559

**Confusion Matrix:**

| |1|2|3|4|
|---|---|---|---|---|
|1| 202967| 15691 | 1957   | 15231 |
|2| 13571 | 205718| 13260  | 6712  |
|3| 2008  | 17728 | 120338 | 21286 |
|4| 17248 | 7860  | 18205  | 326280|

# ResNet34StatsPool 1

- No light tone
- No initials
- 60K train samples
- batch size 64
- SGD(lr=0.01, momentum=0.9)
- ReduceLROnPlateau(patience=4, factor=0.1)

|||||
|---|---|---|---|
|1739 |  272 |   48 |  347|
| 247 | 1968 |  205 |  224|
|  84 |  270 | 1100 |  472|
| 254 |  180 |  238 | 3209|

Epoch 32 Loss 0.8935 Accuracy 0.680 lr 0.000001 acc_val 0.738
