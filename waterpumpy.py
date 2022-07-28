DATA_PATH = '../data/'

import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.merge(pd.read_csv(DATA_PATH+'Tanzanian-Water-Pumps-and-a-Data-Revolution\\train_features.csv'), 
                 pd.read_csv(DATA_PATH+'Tanzanian-Water-Pumps-and-a-Data-Revolution\\train_labels.csv'))
test = pd.read_csv(DATA_PATH+'Tanzanian-Water-Pumps-and-a-Data-Revolution\\test_features.csv')
sample_submission = pd.read_csv(DATA_PATH+'Tanzanian-Water-Pumps-and-a-Data-Revolution\\SubmissionFormat.csv')

# Split train into train & val
train, val = train_test_split(train, train_size=0.80, test_size=0.20, 
                              stratify=train['status_group'], random_state=42)

train.shape, val.shape, test.shape

"""This is a multi-class classification problem. The majority class occurs with 54% frequency."""

train['status_group'].value_counts(normalize=True)

from pandas_profiling import ProfileReport
profile = ProfileReport(train, minimal=True).to_notebook_iframe()

profile

"""Note that longitude has 3% zeros. Some of the locations are at ["Null Island"](https://en.wikipedia.org/wiki/Null_Island) instead of Tanzania."""

import plotly.express as px
px.scatter(train, x='longitude', y='latitude', color='status_group', opacity=0.1)

"""The latitude is approximately zero, but not exactly:"""

train[['longitude', 'latitude']].describe()

train.head()

import numpy as np

def wrangle(X):
    """Wrangle train, validate, and test sets in the same way"""
    
    # Prevent SettingWithCopyWarning
    X = X.copy()
    
    # About 3% of the time, latitude has small values near zero,
    # outside Tanzania, so we'll treat these values like zero.
    X['latitude'] = X['latitude'].replace(-2e-08, 0)
    
    # When columns have zeros and shouldn't, they are like null values.
    # So we will replace the zeros with nulls, and impute missing values later.
    cols_with_zeros = ['longitude', 'latitude']
    for col in cols_with_zeros:
        X[col] = X[col].replace(0, np.nan)
            
    # quantity & quantity_group are duplicates, so drop one
    X = X.drop(columns='quantity_group')
    
    # return the wrangled dataframe
    return X


train = wrangle(train)
val = wrangle(val)
test = wrangle(test)

"""Now the locations look better."""

import plotly.express as px
px.scatter(train, x='longitude', y='latitude', color='status_group', opacity=0.1)

"""By the way, you can also make maps with Plotly Express. For example:"""

# https://zoom.earth/#view=-1.7,30.4,5z/date=2022-07-17,21:30,-4/map=live/overlays=heat,fires,crosshair

# https://plot.ly/python/mapbox-layers/#base-maps-in-layoutmapboxstyle
fig = px.scatter_mapbox(train, lat='latitude', lon='longitude', color='status_group', opacity=0.1)
fig.update_layout(mapbox_style='stamen-terrain')
fig.show()

"""#### Select features"""

# The status_group column is the target
target = 'status_group'

# Get a dataframe with all train columns except the target & id
train_features = train.drop(columns=[target, 'id'])

# Get a list of the numeric features
numeric_features = train_features.select_dtypes(include='number').columns.tolist()

# Get a series with the cardinality of the nonnumeric features
cardinality = train_features.select_dtypes(exclude='number').nunique()

# Get a list of all categorical features with cardinality <= 50
categorical_features = cardinality[cardinality <= 50].index.tolist()

# Combine the lists 
features = numeric_features + categorical_features
print(features)

# Arrange data into X features matrix and y target vector 
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]



"""Here's the documentation for each step in the process:

- https://contrib.scikit-learn.org/category_encoders/onehot.html
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

encoder = ce.OneHotEncoder(use_cat_names=True)
imputer = SimpleImputer()
scaler = StandardScaler()
model = LogisticRegression(max_iter=1000)

X_train_encoded = encoder.fit_transform(X_train)
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_train_scaled = scaler.fit_transform(X_train_imputed)
model.fit(X_train_scaled, y_train)

X_val_encoded = encoder.transform(X_val)
X_val_imputed = imputer.transform(X_val_encoded)
X_val_scaled = scaler.transform(X_val_imputed)
print('Validation Accuracy', model.score(X_val_scaled, y_val))

X_test_encoded = encoder.transform(X_test)
X_test_imputed = imputer.transform(X_test_encoded)
X_test_scaled = scaler.transform(X_test_imputed)
y_pred = model.predict(X_test_scaled)

"""#### Pipelines can help you write more concise code with fewer errors!

Let's rewrite the code cell above, but with a pipeline.

https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
"""

from sklearn.tree import DecisionTreeClassifier

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True),
    SimpleImputer(strategy = 'mean'),
    DecisionTreeClassifier(random_state=42)
)

# fit on train 
pipeline.fit(X_train, y_train)

# score on validation
print('Validation Accuracy', pipeline.score(X_val, y_val))
print('train accuracy', pipeline.score(X_train, y_train))

# predict on test
y_pred = pipeline.predict(X_test)

"""#### Get and plot coefficients

This is slightly harder when using pipelines.

The pipeline doesn't have a `.coef_` attribute. But the model inside the pipeline does. 

So, here's [how to access steps inside a pipeline](https://scikit-learn.org/stable/modules/compose.html#accessing-steps):

> Pipeline’s `named_steps` attribute allows accessing steps by name
"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install graphviz
import graphviz
from sklearn.tree import export_graphviz
# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

model = pipeline.named_steps['decisiontreeclassifier']
encoder = pipeline.named_steps['onehotencoder']
encoded_columns = encoder.transform(X_val).columns

dot_data = export_graphviz(model,
                          out_file=None,
                          max_depth=3,
                          feature_names=encoded_columns,
                          class_names=model.classes_,
                          impurity=False,
                          filled=True,
                          proportion=True,
                          rounded=True)
display(graphviz.Source(dot_data))
# coefficients = pd.Series(model.coef_[0], encoded_columns)
# plt.figure(figsize=(10,30))
# coefficients.sort_values().plot.barh(color='grey');

"""### Plot the decision tree"""

# Plot tree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
import graphviz
from sklearn.tree import export_graphviz

model = pipeline.named_steps['decisiontreeclassifier']
encoder = pipeline.named_steps['onehotencoder']
encoded_columns = encoder.transform(X_val).columns

dot_data = export_graphviz(model, 
                           out_file=None, 
                           max_depth=3, 
                           feature_names=encoded_columns,
                           class_names=model.classes_, 
                           impurity=False, 
                           filled=True, 
                           proportion=True, 
                           rounded=True)   
display(graphviz.Source(dot_data))

"""### Reduce complexity of the decision tree

We can use the `min_samples_leaf` parameter to reduce model complexity.

It's explained in [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

Also in [A Visual Introduction to Machine Learning, Part 2](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/):

> Models can be adjusted to change the way they fit the data. These 'settings' are called [hyper]parameters. An example of a decision-tree [hyper]parameter is the _minimum node size,_ which regulates the creation of new splits. A node will not split if the number of data points it contains is below the minimum node size.
"""

# increase minimum node size also known as minimum samples per leaf
# reduce model complexity 

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True),
    SimpleImputer(strategy='mean'),
    DecisionTreeClassifier(min_samples_leaf=20, random_state=42)
)

pipeline.fit(X_train, y_train)

# score oon train, and validation
print('Train Accuracy', pipeline.score(X_train, y_train))
print('Validation Accuracy', pipeline.score(X_val, y_val))

"""## Overview

Linear models have coefficients, but trees don't. Instead, they have "feature importances."

Feature importances are always positive numbers, they don't have a sign. Feature importance measures how early & often a feature is used for the tree's "branching" decisions. 

Like most predictive modeling tools and techniques, feature importances are useful, but have tradeoffs, make assumptions, and can be misinterpreted.

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

model = pipeline.named_steps['decisiontreeclassifier']

encoder = pipeline.named_steps['onehotencoder']
encoded_columns = encoder.transform(X_val).columns
importances = pd.Series(model.feature_importances_, encoded_columns)
importances.sort_values()

plt.figure(figsize=(10, 30))
importances.sort_values().plot.barh()

"""### Example 1: Predict which waterpumps are functional, just based on location
(2 features, non-linear, feature interactions, classification)

### Compare a Logistic Regression with 2 features, longitude & latitude ...
"""

train_location = X_train[['longitude', 'latitude']].copy()
val_location = X_val[['longitude', 'latitude']].copy()

# With just long & lat, a Logistic Regression can't beat the majority classifier baseline

lr = make_pipeline(
    SimpleImputer(),
    LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=-1)
)

lr.fit(train_location, y_train)
print('Logistic Regression:')
print('Train Accuracy', lr.score(train_location, y_train))
print('Validation Accuracy', lr.score(val_location, y_val))

"""### ... versus a Decision Tree Classifier with 2 features, longitude & latitude

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

# With the same 2 features, the Decision Tree's validation accuracy is 
# almost 10 percentage points better

from sklearn.tree import DecisionTreeClassifier

dt = make_pipeline(
    SimpleImputer(), 
    DecisionTreeClassifier(max_depth=16, random_state=42)
)

dt.fit(train_location, y_train)
print('Decision Tree:')
print('Train Accuracy', dt.score(train_location, y_train))
print('Validation Accuracy', dt.score(val_location, y_val))

"""### Visualize the logistic regression predictions"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import itertools
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns

def pred_heatmap(model, X, features, class_index=-1, title='', num=100):
    """
    Visualize predicted probabilities, for classifier fit on 2 numeric features
    
    Parameters
    ----------
    model : scikit-learn classifier, already fit
    X : pandas dataframe, which was used to fit model
    features : list of strings, column names of the 2 numeric features
    class_index : integer, index of class label
    title : string, title of plot
    num : int, number of grid points for each feature
    """
    feature1, feature2 = features
    min1, max1 = X[feature1].min(), X[feature1].max()
    min2, max2 = X[feature2].min(), X[feature2].max()
    x1 = np.linspace(min1, max1, num)
    x2 = np.linspace(max2, min2, num)
    combos = list(itertools.product(x1, x2))
    y_pred_proba = model.predict_proba(combos)[:, class_index]
    pred_grid = y_pred_proba.reshape(num, num).T
    table = pd.DataFrame(pred_grid, columns=x1, index=x2)
    plot_every_n_ticks = int(floor(num/4))
    sns.heatmap(table, # vmin=0, vmax=1, 
                xticklabels=plot_every_n_ticks, 
                yticklabels=plot_every_n_ticks)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(title)
    plt.show()

pred_heatmap(lr, train_location, features=['longitude', 'latitude'], 
             class_index=0, title='Logstic Regression, predicted probability, "functional"')

pd.Series(lr.named_steps['logisticregression'].coef_[0], 
          train_location.columns)

"""### Visualize the decision tree predictions"""

pred_heatmap(dt, train_location, features=['longitude', 'latitude'], 
             class_index=0, title='Decision Tree, predicted probability, "functional"')

"""### How does a tree grow? Branch by branch!"""

from IPython.display import display, HTML
import graphviz
from sklearn.tree import export_graphviz

for max_depth in [1,2,3]:
    
    # Fit decision tree
    dt = make_pipeline(
        SimpleImputer(), 
        DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    )
    dt.fit(train_location, y_train)
    
    # Display depth & scores
    display(HTML(f'Max Depth {max_depth}'))
    display(HTML(f'Train Accuracy {dt.score(train_location, y_train):.2f}'))
    display(HTML(f'Validation Accuracy {dt.score(val_location, y_val):.2f}'))
    
    # Plot heatmap of predicted probabilities
    pred_heatmap(dt, train_location, features=['longitude', 'latitude'], 
                 class_index=0, title='Predicted probability, "functional"')
    
    # Plot tree
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    dot_data = export_graphviz(dt.named_steps['decisiontreeclassifier'], 
                               out_file=None, 
                               max_depth=3, 
                               feature_names=train_location.columns,
                               class_names=dt.classes_, 
                               impurity=False, 
                               filled=True, 
                               proportion=True, 
                               rounded=True)   
    display(graphviz.Source(dot_data))

for max_depth in range(4,21):
    
    # Fit decision tree
    dt = make_pipeline(
        SimpleImputer(), 
        DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    )
    dt.fit(train_location, y_train)
    
    # Display depth & scores
    display(HTML(f'Max Depth {max_depth}'))
    display(HTML(f'Train Accuracy {dt.score(train_location, y_train):.2f}'))
    display(HTML(f'Validation Accuracy {dt.score(val_location, y_val):.2f}'))
    
    # Plot heatmap of predicted probabilities
    pred_heatmap(dt, train_location, features=['longitude', 'latitude'], 
                 class_index=0, title='Predicted probability, "functional"')

"""# Sources

- A Visual Introduction to Machine Learning
  - [Part 1: A Decision Tree](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
  - [Part 2: Bias and Variance](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
- [Decision Trees: Advantages & Disadvantages](https://christophm.github.io/interpretable-ml-book/tree.html#advantages-2)
- [How a Russian mathematician constructed a decision tree — by hand — to solve a medical problem](http://fastml.com/how-a-russian-mathematician-constructed-a-decision-tree-by-hand-to-solve-a-medical-problem/)
- [How decision trees work](https://brohrer.github.io/how_decision_trees_work.html)
- [Let’s Write a Decision Tree Classifier from Scratch](https://www.youtube.com/watch?v=LDRbO9a6XPU) 
- [Random Forests for Complete Beginners: The definitive guide to Random Forests and Decision Trees](https://victorzhou.com/blog/intro-to-random-forests/)
"""

