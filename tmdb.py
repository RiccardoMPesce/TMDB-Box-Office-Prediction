#%% [markdown]
# TMDB: Box Office Prediction
# **[Link to the challenge](https://www.kaggle.com/c/tmdb-box-office-prediction)**
#
# ## Description
# We're going to make you an offer you can't refuse: a Kaggle competition!
#
# In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate."
#
# In this competition, you're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release.
#
# Join in, "make our day", and then "you've got to ask yourself one question: 'Do I feel lucky?'"

#%% [markdown]
# ## Exploratory Data Analysis
#
# Without further ado, let's first view the dataset, print out the predictors columns and try to get the first insight of the data.

#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# Using seaborn to plot
import seaborn as sns
# Using os.path.join() to create system-independent paths
import os

# Defining where our data is located (for more than one folders, use a list)
DATA_FOLDER = "data"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Creating file paths
train_path = os.path.join(DATA_FOLDER, TRAIN_FILE)
test_path = os.path.join(DATA_FOLDER, TEST_FILE)

# Loading data with pandas (parsing date columns appropriately)
train_df = pd.read_csv(train_path, parse_dates=["release_date"])
test_df = pd.read_csv(test_path, parse_dates=["release_date"])

with pd.option_context("display.max_columns", None):
    display(train_df.head())

# Printing shape and columns of the data
print("There are " + str(train_df.shape[0]) + " observations and " + str(train_df.shape[1]) + " features (including the response).")
print("Columns are: ")
for col in train_df.columns:
    print(col)

#%% [markdown]
# So our dataset is composed of 3000 observations (id of the predictions will start from 3001, so we'll define it next).
#
# #### Dropping useless columns
#
# We can easily guess that the columns *id*, *title*, *homepage*, *imdb_id*, *original_title*, *overview*, *poster_path* and *tagline* won't contribute to our prediction as they don't explain any variance of the dataset, being just additional informations. Therefore, they will be dropped.


#%%
cols_to_drop = ["id", "title", "homepage", "imdb_id", "original_title", "overview", "poster_path", "tagline"]

train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())