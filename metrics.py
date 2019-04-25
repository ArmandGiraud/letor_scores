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
    if y_pred == []:
        return 0
    
    if k > len(y_pred):
        k = len(y_pred)
    
    y_pred = y_pred[:k]

    precision = len(set(y_pred).intersection(y_true))/len(y_pred)

    return precision

def mean_reciprocal_rank(y_pred, y_true):
    """MRR: inverse du rang du premier résultat pertinent
        y_pred: list of documents id predicted by the system
        y_true: list of documents expected by human evaluator
    """
    if y_pred == []: # return 10 if document is empty 
        return 1/100

    res = 1/100 # if document not found give 1/100
    for i, pred in enumerate(y_pred):
        if pred in y_true:
            res = 1/(i + 1)
            break
    return res

def find_recall_k(y_pred, y_true, k):
    """recall: qelle proportion de documents pertinents en regardant seulement les k premiers documents.
    y_pred: list of documents id predicted by the system
    y_true: list of documents expected by human evaluator"""
    if y_pred == []:
        return 0
    res = 0
    nb_relevant = len(y_true)
    if k > len(y_pred):
        k = len(y_pred)
        
    relevant_found = len(set(y_pred[:k]).intersection(y_true))
    return relevant_found/nb_relevant

def discounted_cumulative_gain(y_score, y_true, y_pred, k):
    """y_score is a dictionnary {"doc_id":"score"} of documents assigned as relevant y humans with the associated scores
       k = maximum index to be scored

       /!\ to be valid, dcg should have results lists of same length bbetween requests"""
    if y_pred == []:
        return 0
    if k > len(y_pred):
        k = len(y_pred)
    
    y_scoring = []
    for y in y_pred:
        score = y_score.get(y)
        if score is None: # if the predicted document is not in the array of humanly scored documents
            score = 0
        y_scoring.append(score)
    
    # compute dcg
    def compute_dcg(y_scoring, k):
        dcg = []
        for i, pred in enumerate(y_scoring[:k]):
            i+=1
            dcg.append(pred/(np.log2(i)+1))
        return np.sum(dcg)
    # compute Ideal DCG
    # ideal DCG: the best score that could have been obtained given the relevant document list
    # i. e. : the most relevant documents, ordered by relevance.

    dcg = compute_dcg(y_scoring, k)

    ideal_scores = []
    for i in y_true:
        score = y_score.get(i)
        if not score: # if we don't have a score for y true
            y_score[i]
            raise ValueError("the true document does not have score in y_score")
        ideal_scores.append(score)

    ideal_scores = sorted(ideal_scores, reverse=True)

    ideal_dcg = compute_dcg(ideal_scores, k)
    return dcg/ideal_dcg

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
        return mean_reciprocal_rank(y_pred, y_true)
    elif method == "all":
        return {
            "precision" : find_precision_k(y_pred, y_true, k),
            "recall" : find_recall_k(y_pred, y_true, k),
            "dcg" : discounted_cumulative_gain(y_score, y_true, y_pred, k),
            "mrr": mean_reciprocal_rank(y_pred, y_true)
        }
    else:
        raise ValueError('method shpuld be one of ["precision", "recall", "dcg", "mrr", "all"]')


if __name__ == "__main__":
    # no prediction should result in 0

    y_true = ["a","b","c","d"]
    y_pred = []
    y_score = {"a": 4, 
            "b": 2,
            "c": 2,
            "d": 1}
    k = 3

    assert discounted_cumulative_gain(y_score, y_true, y_pred, k) == 0
    assert find_precision_k(y_pred, y_true, k) == 0
    assert find_recall_k(y_pred, y_true, k) == 0
    assert mean_reciprocal_rank(y_pred, y_true) == 0.01

    # no relevant document in prediction should result in 0

    y_true = ["a","b","c","d"]
    y_pred = ["e","f","g","h"]
    y_score = {"a": 4, 
            "b": 2,
            "c": 2,
            "d": 1}
    k = 3
    assert discounted_cumulative_gain(y_score, y_true, y_pred, k) == 0
    assert find_precision_k(y_pred, y_true, k) == 0
    assert find_recall_k(y_pred, y_true, k) == 0
    assert mean_reciprocal_rank(y_pred, y_true) == 0.01

    # more documents than k should have identical results

    y_true = ["a","b","c","d"]
    y_pred = ["a","f","c","h"]
    y_pred_long = ["a","f","c","h", "d", "b"]

    y_score = {"a": 4, 
            "b": 2,
            "c": 2,
            "d": 1}

    k = 3
    assert discounted_cumulative_gain(y_score, y_true, y_pred, k) == discounted_cumulative_gain(y_score, y_true, y_pred_long, k)
    assert find_precision_k(y_pred, y_true, k) == find_precision_k(y_pred_long, y_true, k)
    assert find_recall_k(y_pred, y_true, k) == find_recall_k(y_pred_long, y_true, k)
    assert mean_reciprocal_rank(y_pred, y_true) == mean_reciprocal_rank(y_pred_long, y_true)

    # assert better ranking give better scores...

    y_true = ["a","b","c","d"]
    y_pred_good = ["a","b","d"]
    y_pred_bad = ["e","a","b","d"]

    y_score = {"a": 4, 
            "b": 2,
            "c": 2,
            "d": 1}

    assert discounted_cumulative_gain(y_score, y_true, y_pred_good, k) > discounted_cumulative_gain(y_score, y_true, y_pred_bad, k)
    assert find_precision_k(y_pred_good, y_true, k) > find_precision_k(y_pred_bad, y_true, k)
    assert find_recall_k(y_pred_good, y_true, k) > find_recall_k(y_pred_bad, y_true, k)
    assert mean_reciprocal_rank(y_pred_good, y_true) > mean_reciprocal_rank(y_pred_bad, y_true)

    # assert too high k causes no error

    k = 5
    assert discounted_cumulative_gain(y_score, y_true, y_pred_good, k)
    assert find_precision_k(y_pred_good, y_true, k) 
    assert find_recall_k(y_pred_good, y_true, k)
    assert mean_reciprocal_rank(y_pred_good, y_true)

    # assert ideal DCG is computed correctly: normalization makes effect

    y_true = ["a"] # one good result with value 1 should yield
    y_pred = ["a","b","d"]

    y_score_high = {
        "a": 5,
        "e": 8 # simultaneously assert e has no impact
    }
    y_score_low = {
        "a": 1
    }
    k = 3

    import numpy as np
    A = discounted_cumulative_gain(y_score_high, y_true, y_pred, k)
    B = discounted_cumulative_gain(y_score_low, y_true, y_pred, k)
    np.testing.assert_almost_equal(A, B) # 

    # assert DCG relative scores 

    y_true = ["a", "b"] # one good result with value 1 should yield
    y_pred = ["a","b"]

    y_score_high = {
        "a": 5,
        "b": 3,
        "c": 5
    }
    y_score_low = {
        "a": 3,
        "b": 2,
        "c": 5
    }
    k = 3

    A = discounted_cumulative_gain(y_score_high, y_true, y_pred, k)
    B = discounted_cumulative_gain(y_score_low, y_true, y_pred, k)
    assert A == B == 1

    # assert perfect results:
    #perfect recall: all relevant documents found
    k = 5
    y_true = ["a", "b"] # one good result with value 1 should yield
    y_pred = ["a","e", "f", "g", "b"]

    assert find_recall_k(y_pred, y_true, 5) == 1
    assert find_precision_k(y_pred, y_true, 5) != 1

    # former assertions still valid
    assert find_recall_k([1, 2, 3], [1], k = 1) == 1
    assert find_recall_k([1, 2, 3], [9], k = 1) == 0
    assert find_recall_k([1, 2, 3], [1], k = 3) == 1
    assert find_recall_k([1, 2, 3], [3, 1], k = 1) == 1/2
    assert find_recall_k([1, 2, 3], [3, 1, 4], k = 2) == 1/3

    # perfect precision:
    k = 2
    y_true = ["a","e", "f", "g", "b"] # one good result with value 1 should yield
    y_pred = ["a", "g"]

    assert find_precision_k(y_pred, y_true, 5) == 1
    assert find_recall_k(y_pred, y_true, 5) != 1

    assert find_precision_k([1,2,3], [1], 3) == 1/3
    assert find_precision_k([1], [1], 3) == 1
    assert find_precision_k([1, 2], [2, 1], 3) == find_precision_k([2, 1], [1, 2], 3) == 1

    # assert dcg  is not messed by additional results

    y_true = ["a", "w"] # one good result with value 1 should yield
    y_pred_good = ["d", "a","w"]
    y_pred_bad = ["d","w","a"]

    y_score = {"a": 5, "w":3}

    assert discounted_cumulative_gain(y_score, y_true, y_pred_good, k) > discounted_cumulative_gain(y_score, y_true, y_pred_bad, k)


    y_true = ["a"] # one good result on first place should be ideal
    y_pred = ["a","b","d"]

    y_score = {"a": 5}
    assert discounted_cumulative_gain(y_score, y_true, y_pred, k) == 1


    # assert y_true is sorted

    y_true_unsorted = ['a', "c", "b"]
    y_true_sorted = ['a', "b", "c"]
    y_pred = ["a", "b", 'w']
    y_score = {"a": 3, 
            "b": 2,
            "c": 1}

    assert discounted_cumulative_gain(y_score, y_true_sorted, y_pred, k) == discounted_cumulative_gain(y_score, y_true_unsorted, y_pred, k)

    # mean reciprocal_rank
    assert mean_reciprocal_rank([1, 2, 3], [1]) == 1
    assert mean_reciprocal_rank([1, 2, 3], [4]) == 0.01
    assert mean_reciprocal_rank([1, 2, 3], [2]) == 1/2
    assert mean_reciprocal_rank([1, 2, 3], [1, 2]) == 1
    assert mean_reciprocal_rank([1, 2, 3], [4, 5, 6]) == 0.01


    # Example:
    # pour la requête "solde de tout compte" on obtient les docs y_pred = [1, 2, 3] les docs pertinents sont dans l'ordre y_true = [7, 2, 3], 
    # avec des scores de pertinences (définies par le métier) y_scores = [0, 3, 1]
    # on calcule la performance de la requête grâce à:
    # multi_score(y_pred, y_true, y_scores)
    # on obtient le score de toutes les requêtes grâce à lz moyenne des scores par reqiêtes