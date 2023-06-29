'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
import random
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    
    
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        #if _testRatings[idx][1] < max_items:
            (hr,ndcg) = eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)      
    return (hits, ndcgs)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, map_item_score.get)
   
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def getPrecision_Recall(user_id,  users_train_items , num_items):
    users_relevant_items = []
    all_items = set(list(range(num_items))) # set of all items
    users_non_relevant_items = []
    for x in _testRatings:
        users_relevant_items.append(x[1]) # relevant items to each user
        
    users_non_relevant_items.extend(list(all_items.difference(set(users_relevant_items).union(set(users_train_items)))))

    sample_size = len(users_relevant_items) 
    test_items_sampled = random.sample(users_non_relevant_items, sample_size)
    test_items_sampled.extend(users_relevant_items)

    user_ids = [user_id] * len(test_items_sampled)    
    predictions = _model.predict([np.array(user_ids), np.array(test_items_sampled)], 
                                 verbose=0)  # make predictions on all items
        
    ranklist = [ (x,y) for x,y in zip(test_items_sampled,predictions)] # sort items and consider topk rated items
    ranklist.sort(key = lambda x: x[1],reverse=True)
    # we will need to consider _K = to 5,10,20 here
    ranklist = ranklist[:_K]
    recommended_items = _K
    recommended_relevant_items = 0
    for i in range(_K):
        item = ranklist[i][0]
        rating = ranklist[i][1]
        if item in users_relevant_items:
            recommended_relevant_items += 1
    precision = recommended_relevant_items / recommended_items
        
    return precision



def get_recommendations(user_id, K = 20, num_items = 1682):
    global _testRatings
    global _model
    
    user_rated_items = [ x[1] for x in _testRatings if x[0] == user_id ]
    user_nonerated_items = [ x for x in range(num_items) if x not in user_rated_items ]
    items = np.hstack((user_rated_items,user_nonerated_items))
    
    user_id = [user_id] * items.shape[0]

    predictions = _model.predict([np.array(user_id), items], 
                                 verbose=0)


    ratings = [ (x,y) for x,y in zip(items,predictions)]
    ratings.sort(key = lambda x: x[1],reverse=True)
    items_recommended = list(zip(*ratings[:K]))[0]
 
    return items_recommended


def get_recommended_items_distribution(user_id, H,M,T, num_items):
    recommendations = get_recommendations(user_id= user_id, num_items = num_items)
    h = m = t = 0
    for item in recommendations:
            if  str(item) in H:
                h += 1
            elif str(item) in M:
                m += 1
            else:
                t += 1

    recommended_prop = [h,m,t]
    recommended_prop = [item / sum(recommended_prop) for item in recommended_prop]
    return recommended_prop


def User_Popularity_Deviation(user_id, train_prop, H, M, T, num_items):
    recommended_prop = get_recommended_items_distribution(user_id, H,M,T, num_items)
    UPD = jensenshannon(train_prop, recommended_prop)
    return UPD