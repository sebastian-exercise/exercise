import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from abc import ABCMeta, abstractmethod
import heapq

class NotFittedError(ValueError, AttributeError):
    """Exception to raise if recommender is asked for recommendation before being fitted """
    


class Recommender(object):
    """ Common interface for every recommender """
    
    __metaclass__ = ABCMeta
        
    @abstractmethod
    def train(self, ratings):
        """ Trains the recommender on the ratings matrix
        
        Parameters
        ----------
        ratings : csc_matrix
            A number_of_users x number_of_products sparse matrix. 
            If user i rated product j, ratings_i,j should be an int greater than zero 
            indicating the user i rating for product j. Otherwise, ratings_i,j should
            be zero. 
        """
        pass
    
    @abstractmethod
    def score(self, user_ratings, include):
        """ Computes a score for each product in include. 
        
        Parameters
        ----------
        user_ratings : array-like
            n_products array of user ratings, where n_products is the number of 
            products in the ratings matrix
        include : array-like, optional
            products to compute a score for
        """
        pass
    
    @abstractmethod
    def def_when_trained(self):
        """ Returns the list of attributes that are supposed to be defined if the 
            recommender is fitted
            
        """
        pass
     
    def fit(self, ratings):
        """ Stores useful info about the ratings matrix and trains
        
        Parameters
        ----------
        ratings : csc_matrix
            A number_of_users x number_of_products sparse matrix. 
            If user i rated product j, ratings_i,j should be an int greater 
            than zero indicating the user i rating for product j. Otherwise, 
            ratings_i,j should be zero. 
        """
        self.n_products = ratings.shape[1]
        self.train(ratings)
        
    def check_is_fitted(self):
        """ Checks if the recommender is fitted
        
        Checks if the attributes required for the recommender to be considered 
        fitted are defined. In particular, every attribute in the attribute 
        def_when_trained have to be defined. If any of the attributes is not 
        defined, a NotFittedError is raised
        
        """
        
        required = self.def_when_trained() + ['n_products']
        if not all([hasattr(self, attr) for attr in required]):
            raise NotFittedError("Recommender is not fitted yet.")
    
    def recommend(self, user_ratings, n, exclude=[]):
        """ Recommends top n products to user 
        
        Parameters
        ----------
        user_ratings : array-like
            n_products array of user ratings, where n_products is the number of products in
            the ratings matrix
        n : int 
            number of products to recommend
        exclude : array-like, optional
            products to exclude from recommendation
        """
        
        self.check_is_fitted()
        
        # Find out which products scores should be computed for
        include = [i for i in range(0, self.n_products) if i not in exclude]
    
        # Compute scores
        scores = self.score(user_ratings, include)
                
        # Return top n
        nlargest = heapq.nlargest(n, zip(include, scores), key=lambda x:x[1])
        return [r[0] for r in nlargest]
    
    
class PureSVDRecommender(Recommender):
    """ PureSVD recommender
    
    Recommender based on conventional Singular Value Decomposition (SVD) for
    factorizing the ratings matrix into two lower rank matrices. After SVD, 
    each user is represented by an n_factors user factors vector, and each 
    item is represented by an n_factors item factors vector.
    
    Parameters
    ----------
    n_factors : int
        The number of factors
    """
    
    def __init__(self, n_factors):
        self.n_factors = n_factors
        
    def train(self, ratings):
        """ Singular Value Decomposition 
        
        Performs SVD, computing two lower rank matrices from the rankings matrix: 
        1) The user latent factor matrix will be a n_users x n_factors matrix
        2) The product latent factor matrix will be a n_items x n_factors matrix

        Parameters
        ----------
        ratings : csc_matrix
            A number_of_users x number_of_products sparse matrix. 
            If user i rated product j, ratings_i,j is an int greater than zero indicating the 
            user i rating for product j. Otherwise, ratings_i,j is zero. 
        """
        
        _, _, factors = svds(ratings, self.n_factors, return_singular_vectors='vh')
        self.product_factors = factors.transpose()
        
    def def_when_trained(self):
        """ Returns the list of attributes that are supposed to be defined if the 
            recommender is trained: [product_factors]
            
        """
        
        return ['product_factors']
        
    def score(self, user_ratings, include):
        """ Computes a score for each product in include. 
        
        Score for a given product p is computed as follows: 
        1) user_ratings is multiplied by the product latent factor matrix. The result of 
        this operation is the user's latent factor vector. 
        2) User's latent factor vector is multiplied by p's latent factor vector to 
        obtain the score. 
        
        Parameters
        ----------
        user_ratings : array-like
            n_products array of user ratings, where n_products is the number of products in
            the ratings matrix
        include : array-like, optional
            products to compute a score for
        """
    
        included_factors = self.product_factors[include] 
        scores = (user_ratings.dot(self.product_factors)).dot(included_factors.transpose()) 
        return scores
    
    
class RandomRecommender(Recommender):
    """ Random recommender
    
    Recommends random products.
    
    Parameters
    ----------
    n_factors : int
        The number of factors
    """
    
    def __init__(self): 
        pass
        
    def train(self, ratings):
        """ Train does nothing for this recommender 
        
        Parameters
        ----------
        ratings : csc_matrix
            A number_of_users x number_of_products sparse matrix. 
            If user i rated product j, ratings_i,j is an int greater than zero indicating the 
            user i rating for product j. Otherwise, ratings_i,j is zero. 
        """
        pass
    
    def def_when_trained(self):
        """ Returns the list of attributes that are supposed to be defined if the 
            recommender is trained: []
            
        """
        
        return []
        
    def score(self, user_ratings, include):
        """ Computes a score for each product in include. 
        
        Score for a given product p is a random number in the range [0,1)
        
        Parameters
        ----------
        user_ratings : array-like
            n_products array of user ratings, where n_products is the number of products in
            the ratings matrix
        include : array-like, optional
            products to compute a score for
        """
        
        scores = np.random.randn(len(include))
        return scores