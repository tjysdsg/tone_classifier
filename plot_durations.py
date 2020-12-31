import json
import matplotlib.pyplot as plt
import seaborn as sns


align = json.load(open('align.json'))
align = {int(k): v for k,v in align.items()}

durs = [dur for k, v in align.items() for _, _, _, dur in v ]
sns.distplot(durs)
plt.title('Distribution of the duration of triphones')
plt.savefig('durations.png')