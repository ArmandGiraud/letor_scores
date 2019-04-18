import numpy as np
#from sklearn.metrics import label_ranking_average_precision_score # this should be applied on lists of lists

#  - precision at k: quelle proportion de resultat pertinents dans les k premiers documents.
#  - recall at k: quelle proportion de documents pertinents en regardant seulement les k premiers documents.
#  - average precision : multiplicative mean between precision and recall at k for k = 1 à n (resemble au fscore)
#  - MAP: moyenne des average precisions
#  - DCG : takes into account the relevance factor given by an user (discounted : pénalisation logarithmique)
#  - nDCG : normalized Dicounted cumulative gain 
#  - mean reciprocal rank (inverse du rang du premier résultat pertinent)

def find_precision_k(y_pred, y_true, k):
    """
     precision at k: quelle proportion de resultat pertinents dans les k premiers documents.

    Description reference:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.

    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]
        k           - a number of top element to consider

    Returns:
        precision   - a score
    """
    
    precision = 0.0
    relevant = 0.0
    for i, value in enumerate(y_true[:k]):
        if value == y_pred[i]:
            relevant += 1.0
    precision = relevant/k

    return precision

def mean_reciprocal_rank(y_pred, y_true):
    """MRR: inverse du rang du premier résultat pertinent
        y_pred: list of documents id predicted by the system
        y_true: list of documents expected by human evaluator
    """
    res = 0
    for i, pred in enumerate(y_pred):
        if pred in y_true:
            res = 1/(i + 1)
            break
    return res

def find_recall_k(y_pred, y_true, k):
    """recall: qelle proportion de documents pertinents en regardant seulement les k premiers documents.
    y_pred: list of documents id predicted by the system
    y_true: list of documents expected by human evaluator"""
    res = 0
    nb_relevant = len(y_true)
    if k > len(y_pred):
        raise ValueError("k: longer than nb of results in y_pred")
        
    relevant_found = len(set(y_pred[:k]).intersection(y_true))
    return relevant_found/nb_relevant

def discounted_cumulative_gain(y_pred, k):
    """y_pred is the list of relevance scores of results (0 if not scored by humans)
       k = maximum index to be scored
       /!\ to be valid, dcg should have results lists of same length"""
    res = []
    for i, pred in enumerate(y_pred[:k]):
        i+=1
        res.append(pred/(np.log2(i)+1))
    return np.mean(res)

def multi_score(y_pred_id, y_true_id, y_pred_scores):
    print("recall: at 1", find_recall_k(y_pred_id, y_true_id, min(1, len(y_pred_id)))) # min make sure the prediction array is not shorter than the expected results...
    print("recall: at 3", find_recall_k(y_pred_id, y_true_id, min(3, len(y_pred_id))))
    print("recall: at 5", find_recall_k(y_pred_id, y_true_id, min(5, len(y_pred_id))))
    print()
    print("precision: at 1", find_precision_k(y_pred_id, y_true_id, min(1, len(y_pred_id))))
    print("precision: at 3", find_precision_k(y_pred_id, y_true_id, min(3, len(y_pred_id))))
    print("precision: at 5", find_precision_k(y_pred_id, y_true_id, min(5, len(y_pred_id))))


    print("DCG at 1: ", discounted_cumulative_gain(y_pred_scores), 1)
    print("DCG at 3: ", discounted_cumulative_gain(y_pred_scores), 3)
    print("DCG at 5: ", discounted_cumulative_gain(y_pred_scores), 5)

    print("----")
    print("mean reciprocal rank: ", mean_reciprocal_rank(y_pred_id, y_true_id))

def score(y_pred, y_true, y_score, k, method):
    """y_pred: documents id returned by the system sorted from most relevant to least relevant
       y_true: documents id scored by humans sorted from most relevant to least relevant
       y_scores: human scores of documents returned by the system (o if not scored)
       k: maximum rank to be considered
       method: one of ["precision", "recall", "dcg", "mrr", "all"]"""

    if method not in ["precision", "recall", "dcg", "mrr", "all"]:
        raise ValueError('method shpuld be one of ["precision", "recall", "dcg", "mrr"]')

    if method == "precision":
        return find_precision_k(y_pred, y_true, k)
    elif method == "recall":
        return find_recall_k(y_pred, y_true, k)
    elif method == "dcg":
        return discounted_cumulative_gain(y_score, k)
    elif method == "mrr":
        mean_reciprocal_rank(y_pred, y_true, k)
    elif method == "all":
        return {
            "precision" : find_precision_k(y_pred, y_true, k),
            "recall" : find_recall_k(y_pred, y_true, k),
            "dcg" : discounted_cumulative_gain(y_score, k),
            "mrr": mean_reciprocal_rank(y_pred, y_true)
        }
    else:
        raise ValueError('method shpuld be one of ["precision", "recall", "dcg", "mrr", "all"]')


if __name__ == "__main__":
    # mean reciprocal_rank
    assert mean_reciprocal_rank([1, 2, 3], [1]) == 1
    assert mean_reciprocal_rank([1, 2, 3], [4]) == 0
    assert mean_reciprocal_rank([1, 2, 3], [2]) == 1/2
    assert mean_reciprocal_rank([1, 2, 3], [1, 2]) == 1
    assert mean_reciprocal_rank([1, 2, 3], [4, 5, 6]) == 0

    # recall at k:
    assert find_recall_k([1, 2, 3], [1], k = 1) == 1
    assert find_recall_k([1, 2, 3], [9], k = 1) == 0
    assert find_recall_k([1, 2, 3], [1], k = 3) == 1
    assert find_recall_k([1, 2, 3], [3, 1], k = 1) == 1/2
    assert find_recall_k([1, 2, 3], [3, 1, 4], k = 2) == 1/3

    # dcg
    assert discounted_cumulative_gain([0,0,0,0], 10) == 0.0
    assert discounted_cumulative_gain([1,0,0,0], 10) == 0.25
    assert discounted_cumulative_gain([0,1,0,0], 10) == 0.125

    # Example:
    # pour la requête "solde de tout compte" on obtient les docs y_pred = [1, 2, 3] les docs pertinents sont dans l'ordre y_true = [7, 2, 3], 
    # avec des scores de pertinences (définies par le métier) y_scores = [0, 3, 1]
    # on calcule la performance de la requête grâce à:
    # multi_score(y_pred, y_true, y_scores)
    # on obtient le score de toutes les requêtes grâce à lz moyenne des scores par reqiêtes