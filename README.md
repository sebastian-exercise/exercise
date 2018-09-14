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
1. Clone this repository and download the Yelp challenge dataset. Please make sure the code and the data follow this structure:

```
- _<exercise-dir>_/
  - code/
    - contagiousness.ipynb
    - recommendations.ipynb
    - fracc_weekends_prediction.ipynb
    - ...
  - data/
    - yelp_academic_dataset_review.json
    - yelp_academic_dataset_user.json
    - ...
```


