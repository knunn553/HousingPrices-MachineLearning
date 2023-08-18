# %%
!pwd
%cd /Users/kylenunn/Desktop/Machine-Learning

import jupyter, matplotlib, numpy, pandas, scipy, sklearn

!pwd

import os
import tarfile
from six.moves import urllib

Download_Root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
Housing_Path = os.path.join("datasets", "housing")
Housing_URL = Download_Root + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=Housing_URL, housing_path=Housing_Path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
# This below function (and above) is just simply pulling the data set from github using a number of functions
# Obviously you aren't going to know this now but you will in the future
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
    
# Now when you call fetch_housing_data(), it creates a datasets/housing directory in
# your workspace, downloads the housing.tgz file, and extracts the housing.csv from it in
# this directory.

fetch_housing_data()

# Now we are loading the data using Pandas
# In addition, we need to create a function to load the data

def load_housing_data(housing_path=Housing_Path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# This function creates a Pandas Dataframe object consisting of all the data.
# Now we will load the data
import pandas as pd
housing = load_housing_data()
housing.head()

# info() command is essential to get quick and necessary information from the data set.
housing.info()

# float64 = numerical object in Python
#You can find out what categories exist and how many districts belong to each category by using the value_counts() method:
housing["ocean_proximity"].value_counts()

# Let’s look at the other fields. The describe() method shows a summary of the numerical attributes
housing.describe()

%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

!pwd
# The hist() method relies on Matplotlib
%matplotlib inline


# %%
#And important thing to do when the data is finally cleaned and appropriately labeled is to create a test set which is normally about 20% of the entire data samples.
# Creating a test set:
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# We can then use the above function as follows:

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

# To avoid getting different test and training data sets, its important to save the first train and test sets using signal operators.


# %%
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# When code is not running, it is very important to check how far the code is indented along with spacing and capital letters.

# Unfortunately, the housing dataset does not have an identifier column. The simplest solution is to use the row index as the ID:

housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# If you use the row index as a unique identifier, you need to make sure that new data
# gets appended to the end of the dataset, and no row ever gets deleted. If this is not
# possible, then you can try to use the most stable features to build a unique identifier.
# For example, a district’s latitude and longitude are guaranteed to be stable for a few
# million years, so you could combine them into an ID like so:

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
housing = strat_train_set.copy()





# %%
housing.plot(kind="scatter", x="longitude", y="latitude")
# Sometimes if codes aren't loading make sure the above code is all loaded including your library.
# In addition, make sure you aren't starting something totally new to where it needs to be placed into a new cell box
# If all else fails, just make a new cell and try it.
# In the above code, setting the alpha option the a lower mark allows us to see the higher density places.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
#The radius of each circle represents
#the district’s population (option s), and the color represents the price (option c). We
#will use a predefined color map (option cmap) called jet, which ranges from blue
#(low values) to red (high prices)


# %%
# Now we will look for correlations between variables.
# Since the data set is not too large, we can calculate a simple pearsons r correlation using the corr() method/function.
corr_matrix = housing.corr()
# Now we'll see how each attribute correlates with the median housing value:
corr_matrix["median_house_value"].sort_values(ascending=False)
# The correlation coefficient only measures linear correlations.
# Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function, which plots every numerical attribute against every other numerical attribute.
from pandas.plotting import scatter_matrix
# We have to import the scatter_matrix function using pandas library
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# This basically allows us to see the correlation using pictures isntead of numbers. This is great for visual observors like myself.
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)

# %%
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# After assigning these attributes, lets look at the correlation matrix
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
# Now we will prepare the data for machine learning algorithms.
# Building a library of transformation can and should be used in future projects. Its a great idea to save work.
# (note that drop() creates a copy of the data and does not affect strat_train_set):
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# We can't have a. data set with missing values.
# In this scenario, the variable total_bedrooms attribute has missing values and we can fix that in one of the following three ways:
# 1) Get rid of the corresponding districts, 2) Get rid of the whole attribute, 3) Set the values to some value (zero, the mean, the median, etc.).
# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna()methods:
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

# Most machine learning algorithms cannot work with missing features so the above code attempts to fix that problem.
# We can either get rid of the whole attribute/column, change the missing values to other values (e.g. the mean or median of the data set.)


# %%
# Its important that we save the mean/median filler value so that we can use it in the test set.
# Its also important so we can use the value in say an unserpervised algorithm.
# Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# Dropping ocean proximity attribute and creating a copy of the data because SimpleImputer can only be used for numerical data...
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

# %%
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# %%
# Its important to convert text to numbers becuase that's what computers are better at recognizing, for now..
# In this dataset, we will convert the ocean proximity values into numerical variables.
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# %%
ordinal_encoder.categories_
# Converting ocean proximity attribute into a one hot vector
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# %%
## ONE OF THE MOST IMPORTANT TRANSFORMATIONS WE NEED TO APPLY TO THE DATA IS FEATURE SCALING ####
# Standardization and Normalization are different.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

# Now we can train and evaluate on the training set.
# First, we will run a linear regression model:

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# Let's try this out on a few instances in the training set.
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# Lets measure this regression models RSME compared to the training set using Scikit Learn's mean_square_error function.
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# The results are poor, maybe in due part becuase the model was underfit or not complex enough.
# We will now train a DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse