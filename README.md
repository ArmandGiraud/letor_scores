# Learning to Rank scores api
a flask minimal api to score a ranking system against human relevance scores



## Usage
### Deploy
```bash
sudo docker-compose up --build -d
```
### Endpoint /api/score

the score endpoint enables to score a single request, to evaluate the system on the list of requests, just compute the average of the returned scores 
```python
import requests

params = {
    "y_pred" : ["a", "b", "c", "w", "k","e"], # y_pred (array): list of documents id predicted by the system
    "y_true" : ["a", "b", "c","e"], # y_true (array): documents id scored by humans sorted from most relevant to least relevant
    "y_score" : {
        "a":5,
        "b":3,
        "c":2
        "e":2
            }, # y_score is a dict {"doc_id":"score"} of documents assigned as relevant y humans with the associated scores (ordered internally)
    "method" : "all", # one of ["precision", "recall", "dcg", "mrr", "all"]
    "k": 3 # k integer preferaly =< to length(y_pred)
}

r = requests.post("0.0.0.0:4545/api/score", json = params)
r
>>> 200

r.json()
>>> {'dcg': 2.660558421703625, 'mrr': 1.0, 'precision': 1.0, 'recall': 0.75}

```

## Metrics:
for a given request:

- [x] [precision at k](https://en.wikipedia.org/wiki/Precision_and_recall): quelle proportion de resultat pertinents dans les k premiers documents.
- [x] [recall at k](https://en.wikipedia.org/wiki/Precision_and_recall): quelle proportion de documents pertinents en regardant seulement les k premiers documents.
- [x] [DCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) : takes into account the relevance factor given by an user (discounted : pénalisation logarithmique)
- [x] [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) mean reciprocal rank (inverse du rang du premier résultat pertinent)
- [x] [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) normalized Dicounted cumulative gain 
- [ ] [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score) : multiplicative mean between precision and recall at k for k = 1 à n (resemble au fscore)

