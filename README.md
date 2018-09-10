# Exercise

The exercise is distributed in 3 main Jupyter notebooks (briefly described below):
- contagiousness.ipynb
- recommendations.ipynb
- fracc_weekends_prediction.ipynb

A 4th notebook is included for generating different tables used by the rest of the noteeboks: dataset_genereation.ipynb. **This notebook must be run before any other notebook**.

## contagiousness.ipynb
Explores the hypothesis of a contagious effect between friends considering reviewed businesses.

## recommendations.ipynb
Explores the possibility of building a collaborative filtering recommender based on the available reviews table.

## fracc_weekends_prediction.ipynb
Explores the possibility of predicting the fraction of checkins a business receives during weekends.

# Instructions
The easiest way to reproduce the results is using virtual enviroments. Create a new virtualenv and install standard packages via the command:  
*python -m pip install --user numpy scipy matplotlib ipython jupyter pandas seaborn*

Clone this repository, change the current directory to the one where the repository was saved and launch Jupyter via the command *jupyter notebook*. 

The code assumes the different .json files from the challenge dataset are saved in a folder *data*, located at the same level as the folder containing the notebooks. For example, a proper hierarchy would be:
<some path>/code/contagiousness.ipynb
... /recommendations.ipynb
                /fracc_weekends_prediction.ipynb
                ...
           /data/yelp_academic_dataset_review.json
                /yelp_academic_dataset_user.json
                ...

