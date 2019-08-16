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
pred_id = 3001
cols_to_drop = ["id", "title", "homepage", "imdb_id", "original_title", "overview", "poster_path", "tagline"]

train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())

#%% [markdown]
# #### Transforming columns whose elements are dict
#
# Some columns have dicts as values. Such dicts contains different informations, but we want to just retain the identifying one (in most, this is the *name*). For this reason, we are going to map these column values to the extracted name from the dicts.
#
# For such objective, we will use `pandas`'s method `map()`. We map to the *name* value of the dicts of the column *genres*. For *spoken_languages*, we will map to the keys *iso_639_1*'s values.
#
# To avoid potential malfunctionings, we fill NA values of non numerical features with the class "None", and later, during a further step, we will appropiately choose the right class for such missing features.

#%%
import ast
import math

cat_cols = train_df.select_dtypes(exclude="number").columns

train_df[cat_cols] = train_df[cat_cols].fillna("None")
test_df[cat_cols] = test_df[cat_cols].fillna("None")

train_df["genres"] = train_df["genres"].apply(lambda x: ast.literal_eval(x)[0]["name"] if x != "None" else "None")
test_df["genres"] = test_df["genres"].apply(lambda x: ast.literal_eval(x)[0]["name"] if x != "None" else "None")

train_df["spoken_languages"] = train_df["spoken_languages"].apply(lambda x: ast.literal_eval(x)[0]["iso_639_1"] if x != "None" else "None")
test_df["spoken_languages"] = test_df["spoken_languages"].apply(lambda x: ast.literal_eval(x)[0]["iso_639_1"] if x != "None" else "None")

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())

#%% [markdown]
# #### What should we do about features that could potentially span into lots of indicator variables?
# Initially, we can think that the feature *Keywords* is an important one, since it refers to the feature of a film itself, and so should be very descriptive about it. However, these are just used by imdb so they won't confer so much accuracy to our predictions. Therefore, we are going to drop them, also because mapping every possible keyword to an indicator variable could increase the complexity of our model, and therefore it could overfit the data.
#
# We are also dropping, for the same reason, the features *belongs_to_collection*, *original_language*, *production_companies*, *production_countries*, *cast* and *crew*. These variables, unlike the previous one, could tell us potentially useful informations for inferential analyses. However, they will span out into lot of classes, and during test, there will be some classes, spanned into indicators, that our model hasn't learned, and so, it will be a problem.

#%%
train_df = train_df.drop(["belongs_to_collection", "original_language", "production_companies", "production_countries", "spoken_languages", "Keywords", "cast", "crew"], axis=1)
test_df = test_df.drop(["belongs_to_collection", "original_language", "production_companies", "production_countries", "spoken_languages", "Keywords", "cast", "crew"], axis=1)

# Displaying pruned data
with pd.option_context("display.max_columns", None):
    display(train_df.head())


#%% [markdown]
# #### Encoding
# Now we have to encode our categorical variables. The variable *status* can be encoded with 1/0, while *release_date* can be simply transformed to POSIX timestamp for easy conversion.

#%%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore")

train_df["status"] = train_df["status"].apply(lambda x: 1 if x == "Released" else 0)
test_df["status"] = test_df["status"].apply(lambda x: 1 if x == "Released" else 0)

train_df["release_date"] = train_df["release_date"].apply(lambda x: pd.to_datetime(x).timestamp() if type(x) is "str" else 0)
test_df["release_date"] = test_df["release_date"].apply(lambda x: pd.to_datetime(x).timestamp() if type(x) is "str" else 0)

with pd.option_context("display.max_columns", None):
    display(train_df.head())

enc.fit(train_df)

enc.transform(train_df)
enc.transform(test_df)

with pd.option_context("display.max_columns", None):
    display(train_df.head())


#%%
