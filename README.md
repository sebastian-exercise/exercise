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
- <exercise-dir>/
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
where \<exercise-dir\> is a path of your choice.

2. Run:
```
	docker pull lacroze863/exercise:0.1
```
to pull the docker image from Docker Hub.

3. Run:
```
  docker images
```
to see the table of available images. Copy the value in column _IMAGE ID_ for the image associated to repository _lacroze863/exercise_ (that is, the one that was pulled in the previous step)

4. Change your current directory to \<exercise-dir\> (the directory chosen in step 1)

5. Run:
```
  docker run -p <port>:8888 -v ${PWD}:/exercise <image-id>
```
to lunch the docker container, where:
- \<port\> is a free port of your choice
- \<image-id\> is the image id copied in the previous step.

6. You will a message from Jupyter Notebook. It will end with something like:
```
    Copy and paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://<some url>:8888/?token=<token>
```
where \<token\> is a large alphanumeric string. Copy it.

7. Open a web browser an go to the URL: 
```
  localhost:<port>/?token=<token>
```
where:
- \<port\> is the port choosen in step 5
- \<token\> is the token copied in step 6

8. This will show the Jupyter Notebook interface in your browser. Navigate to the directory where the notebooks were saved in step 1:
```
<exercise-dir>/code
```

9. Open and run each cell (from top to bottom) of notebook _dataset_genereation.ipynb_. After this step, you can run the other 3 notebooks in any order (always run cells from top to bottom within each notebook)
