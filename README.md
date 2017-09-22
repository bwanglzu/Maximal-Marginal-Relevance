# Maximal-Marginal-Relevance

MMR for information retrieval.

#### Install

```
git clone project
python setup.py install
```

#### Usage

```python
import pandas as pd
from mmr import mmr

lambda_score = 0.5
initial_ranking = pd.read_csv('../example/example.csv')
print mmr.rank(initial_ranking, lambda_score)
```


