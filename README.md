# Maximal-Marginal-Relevance

MMR for information retrieval.

#### Install

```
git clone project
pip install -r requirements.txt
```

#### Usage

```python
lambda_score = 0.5
initial_ranking = pd.read_csv('../example/example.csv')
print main(initial_ranking, lambda_score)
```


