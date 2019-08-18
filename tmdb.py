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
#
# Before reading the csv with pandas, we save the file with encoding "utf-8".

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
train_df = pd.read_csv(train_path, parse_dates=["release_date"], encoding="utf-8", low_memory=False)
test_df = pd.read_csv(test_path, parse_dates=["release_date"], encoding="utf-8", low_memory=False)

with pd.option_context("display.max_columns", None):
    display(train_df.head())
    display(test_df.head())

#%% [markdown]
# #### Shape of the data
# Let's now print the number of observations and features of our datasets

#%%
print(train_df.shape)
print(test_df.shape)

#%% [markdown]
# Now let's see the explanatory variables.

#%%
for col in train_df.columns:
    print(col)

#%% [markdown]
# So our dataset is composed of 3000 observations (id of the predictions will start from 3001, so we'll define it next).

#%% [markdown]
# #### Dropping useless columns
#
# We gues that features like *popularity* are very descriptive of the variance in the response.
#
# On the contrary, *id*, *title*, *homepage*, *imdb_id*, *original_title*, *overview*, *poster_path* and *tagline* won't contribute to our prediction as they don't contribute to the revenue of the movie in any way. Therefore, they will be dropped.


#%%
pred_id = 3001
cols_to_drop = ["id", "title", "homepage", "imdb_id", "original_title", "overview", "poster_path", "tagline"]

train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())
    display(test_df.head())

#%% [markdown]
# #### Transforming columns whose elements are dict
#
# Some columns have dicts as values. Such dicts contains different informations, but we want to just retain the identifying one (in most, this is the *name*). For this reason, we are going to map these column values to the extracted name from the dicts.
#
# For such objective, we will use `pandas`'s method `map()`.
#
# Some features contain a single dict, others multiple dict: the former will be mapped to a string with the single name, the latter to the string representation of the list of names, and we will process them further in next steps.
#
# To avoid potential malfunctionings, we fill NA values of non numerical features with the class "None", and later, during a further step, we will appropiately choose the right class for such missing features.

#%%
import ast
# Using json.dumps() to have string with "" in our lists
import json

def parse_name(x):
    '''
    This function will parse the "name" property in the relative dict or list of dicts. 
    Returns either a string containing the name or a string representation of a list of all names.
    '''

    if x != "None":
        ls = ast.literal_eval(x)
    else:
        return "None"
    
    if len(ls) == 1:
        return ls[0]["name"]
    else:
        return json.dumps([d["name"] for d in ls])

def parse_iso(x):
    '''
    This function will parse the "iso_*" property in the relative dict or list of dicts. 
    Returns either a string containing the name or a string representation of a list of all iso_*.
    '''

    if x != "None":
        ls = ast.literal_eval(x)
    else:
        return "None"

    if len(ls) == 1:
        return list(ls[0].values())[0]
    else:
        return str([list(d.values())[0] for d in ls])

# Selecting categorical and numerical columns
cat_cols = train_df.select_dtypes(exclude=["number", "datetime"]).columns
num_cols = train_df.select_dtypes("number").columns

# Saving categorical and numerical columns
train_df[cat_cols] = train_df[cat_cols].fillna("None")
test_df[cat_cols] = test_df[cat_cols].fillna("None")

for cat in ["belongs_to_collection", "genres", "production_companies", "Keywords", "cast", "crew"]:
    train_df[cat] = train_df[cat].map(parse_name)
    test_df[cat] = test_df[cat].map(parse_name)

for cat in ["production_countries", "spoken_languages"]:
    train_df[cat] = train_df[cat].map(parse_iso)
    test_df[cat] = test_df[cat].map(parse_iso)

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())
    display(test_df.head())

#%% [markdown]
# #### Reducing multiple occurrences in the lists and turn lists with just one element to strings.

#%%
def unique_vals(x):
    if not x.startswith("[\""):
        return x 
    else:
        ls = ast.literal_eval(x)
        uniq = list(set(ls))
        if (len(uniq) == 1):
            print(uniq[0])
            return uniq[0]
        else:
            return json.dumps(uniq)

for cat in cat_cols:
    print(cat)
    train_df[cat] = train_df[cat].map(unique_vals)
    test_df[cat] = test_df[cat].map(unique_vals)

# Displaying converted data
with pd.option_context("display.max_columns", None):
    display(train_df.head())
    display(test_df.head())

#%% [markdown]
# #### Statistics about the data
# We can see that we have categories that span into lots of classes. What should we do about them?
#
# First, we need to stack the dataframes (by dropping in the first one the response variable) in order to cumulatively see the class distributions.
#
# Then, for each actor, we count the number of movies he starred in. And we describe a pandas series so that we can get all the statistical informations we need to classify an actor as popular or unpopular.
#
# The crew feature will become "n of popular actors", where we specify the number of popular actors starring in that movie, where popular means that an actor has taken part in a number of movies more than a threshold.

#%%
# Concatenating the two dataframes
merged_df = pd.concat([train_df.drop("revenue", axis=1), test_df], ignore_index=True, sort=False)

actor_dict = {}
n_movies = merged_df.shape[0]

for val in merged_df["cast"]:
    if val.startswith("[\""):
        val_ls = [n for n in ast.literal_eval(val)]
    else:
        val_ls = [val]

    for e in val_ls:
        if e in actor_dict.keys():
            actor_dict[e] += 1
        else:
            actor_dict[e] = 1

actor_freq = pd.Series(actor_dict)

actor_freq.describe()

#%% [markdown]
# We can see that there are only a few actors that appeared 
#
