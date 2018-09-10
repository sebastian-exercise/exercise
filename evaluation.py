import numpy as np
from sklearn.model_selection import KFold
from recommender import Recommender

def is_hit(recommender, user_ratings, product, n_non_rated, top_n):
    """ Tests for a hit for product for a user with user_ratings ratings  
    
    Returns True iff product is in the top top_n recommended items by 
    recommender for a user with user_ratings ratings.
    
    Parameters
    ----------
    recommender : Recommender
        fitted recommender 
    user_ratings : ndarray
        n_products vector of ratings
    product : int
        product index for which the hit test is performed
    n_non_rated: int
        the number of unrated items to select along with product
    top_n: int
        the number of items to recommend
    """
    
    n_products = len(user_ratings)
    
    # Remember product product rating
    product_rating = user_ratings[product]
    
    # Select non-rated random products to rank along with product
    non_rated = np.where(user_ratings == 0)[0]
    exclude_non_rated = np.random.choice(non_rated, len(non_rated) - n_non_rated, replace=False)
    user_ratings[product] = 0
    exclude_rated = np.where(user_ratings)[0]
    exclude = np.concatenate([exclude_non_rated, exclude_rated])
    
    # Get top_n ranking
    recommended = recommender.recommend(user_ratings, top_n, exclude)
    
    # Restore product rating
    user_ratings[product] = product_rating
    
    # We have a hit iff product is among the recommended products
    return product in recommended


def average_recall(recommender, ratings, test_rtg_th, negs_hit_test, top_n, users=None, n_fold=10, 
             random_state=0, verbose=False):
    """ Computes averages recall for the fitted recommender on the ratings matrix
        
    Parameters
    ----------
    recommender : Recommender
        fitted recommender 
    ratings : csc_matrix
        n_users x n_products sparse matrix of ratings
    test_rtg_th : int
        the minimum rating value for an item to be considered as rated, for
        the pourpose of being selected for testing
    negs_hit_test : int
        the number of unrated items to select along with the rated item being 
        tested
    top_n : int
        the number of items to recommend for each test
    users : array-like, optional
        if not None, the users to test the recommender on
    n_fold : int, optional, default 10
        the number of folds to user for cross-validation
    random_sate: int, optional, default 0
        seed of random generator, for reproducibility
    verbose: boolean, optional, default False
         verbosity
    """
    
    if users is None:
        users = range(0, ratings.shape[0])
        
    kf = KFold(n_splits=n_fold, random_state=random_state)
    f = 0
    recall_sum = 0
    
    # For each test fold defined by the KFold regime
    for _, test_users in kf.split(users):
        if verbose:
            print('Fold {} out of {}'.format(f, n_fold))
        
        f = f + 1
        tests = 0
        hits = 0
        
        # Train on ratings not in the test fold
        all_users = np.arange(ratings.shape[0])
        train_users = np.where(np.logical_not(np.in1d(all_users, test_users)))[0]
        train_ratings = ratings[train_users, :]
        recommender.fit(train_ratings)
        
        # For each user u in the test fold
        for u in test_users:
            u_ratings = ratings[u, :].toarray()[0]
            
            # For each item i rated by u (such that i has rating >= test_rtg_th)
            rated_by_u = np.where(u_ratings >= test_rtg_th)[0]
            if(rated_by_u.shape[0] > 50):
                rated_by_u = np.random.choice(rated_by_u, 50)
                
            for p in rated_by_u:
                
                # Update the number of tests and hits for the fold 
                tests = tests + 1
                hits = hits + is_hit(recommender, u_ratings, p, negs_hit_test, top_n)
        
        if tests > 0:
            recall_sum = recall_sum + hits / float(tests)
              
    return recall_sum / n_fold