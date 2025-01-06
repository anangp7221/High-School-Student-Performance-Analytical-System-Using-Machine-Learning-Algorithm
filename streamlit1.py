# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

'''
We are analyzing IRIS dataset with k-means and hierachical clustering methods

'''

# +
# %matplotlib inline
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import rcParams
rcParams['figure.figsize'] = 9, 8  # set plot size

from subprocess import check_output

# Instead of loading the Iris dataset with load_iris(), use pd.read_csv() to read your dataset
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Assuming your dataset is in a CSV file named 'your_dataset.csv'
# If the file is in a different location, specify the correct file path
file_path = 'data_code_obj_2381_target.csv'

# Use pd.read_csv() to read the CSV file and create a DataFrame
df = pd.read_csv('D:/Documents/data_code_obj_2381_target.csv', index_col=[0],error_bad_lines=False, sep=";")

# Display the first few rows of the DataFrame to check if the data is loaded correctly
df.head()
# -

# Display DataFrame information
st.write("DataFrame Information:")
st.dataframe(df.info())

# Display first 90 rows
st.write("First 90 Rows:")
st.dataframe(df.head(90))

# Display transposed summary statistics
st.write("Transposed Summary Statistics:")
st.dataframe(df.describe().T)

# +
column_drop=['tgl_lahir', 'current_date', 'rata-nilai'
            ]

df.drop(columns=column_drop, inplace=True)

df.info()

# +
#Load the Dataset
#import pandas as pd
#from sklearn.datasets import load_iris
#data = load_iris()
#df = pd.DataFrame(data.data, columns=data.feature_names)
#df['target'] = data.target

# +
pd.set_option("display.max_rows", None, "display.max_columns", None) 

df.describe().T
# -

df.head(90)

# Display DataFrame information
st.write("DataFrame Information:")
st.dataframe(df.info())

# Display first 90 rows
st.write("First 90 Rows:")
st.dataframe(df.head(90))

# Display transposed summary statistics
st.write("Transposed Summary Statistics:")
st.dataframe(df.describe().T)

df.describe().T

df.describe(include=['int64']).T

# +
#from sklearn.preprocessing import LabelEncoder

#label_encoder = LabelEncoder()
#df['id'] = label_encoder.fit_transform(df['id'])
# -

#non_numeric_columns = ['id']  # Add the names of any non-numeric columns here
#for col in non_numeric_columns:
    #df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.dtypes)


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# +
#create columns yang digunajan saebagai learning habit
column=['class']

df_target=df[column]

st.write("DataFrame Head and Shape:")
st.dataframe(df_target.head())
st.write("DataFrame Shape:", df_target.shape)
# -

df.head()

df_data=df.copy()

df_data.drop(columns='class', inplace=True)

df_data.head()

scaled_features = StandardScaler().fit_transform(df_data)
df_standarscalar = pd.DataFrame(scaled_features, index=df_data.index, columns=df_data.columns)

df_standarscalar.head()

df_target.head()

from sklearn.preprocessing import LabelEncoder

# +
#from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_target['class'] = label_encoder.fit_transform(df_target['class'])

# +
# k-means cluster analysis for 1-15 clusters                                              
from scipy.spatial.distance import cdist
clusters=range(1,15)
meandist=[]

# loop through each cluster and fit the model to the train set
# generate the predicted cluster assingment and append the mean 
# distance my taking the sum divided by the shape
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(df_standarscalar)
    clusassign=model.predict(df_standarscalar)
    meandist.append(sum(np.min(cdist(df_standarscalar, model.cluster_centers_, 'euclidean'), axis=1))
    / df_standarscalar.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method') 
# pick the fewest number of clusters that reduces the average distance
# If you observe after 3 we can see graph is almost linear
# -

# Here we are just analyzing if we consider 2 cluster instead of 3 by using PCA 
model3=KMeans(n_clusters=2)
model3.fit(df_standarscalar) # has cluster assingments based on using 2 clusters
clusassign=model3.predict(df_standarscalar)
# plot clusters
''' Canonical Discriminant Analysis for variable reduction:
1. creates a smaller number of variables
2. linear combination of clustering variables
3. Canonical variables are ordered by proportion of variance accounted for
4. most of the variance will be accounted for in the first few canonical variables
'''
from sklearn.decomposition import PCA # CA from PCA function
pca_2 = PCA(2) # return 2 first canonical variables
plot_columns = pca_2.fit_transform(df_standarscalar) # fit CA to the train dataset
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,) 
# plot 1st canonical variable on x axis, 2nd on y-axis
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show() 
# close or overlapping clusters idicate correlated variables with low in-class variance 
# but not good separation. 2 cluster might be better.

# +
# calculate full dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# generate the linkage matrix
Z = linkage(df_standarscalar, 'ward')

# set cut-off to 150
max_d = 7.08                # max_d as in max_distance

plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()

# +
# calculate full dendrogram for 50
from scipy.cluster.hierarchy import dendrogram, linkage

# generate the linkage matrix
Z = linkage(df_standarscalar, 'ward')

# set cut-off to 50
max_d = 7.08                # max_d as in max_distance

plt.figure(figsize=(25, 10))
plt.title('Student Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=50,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()
# -

# visualisasi:
#     sumber:https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

#from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier()
#iris = load_iris()

clf = clf.fit(df_data, df_target)
tree.export_graphviz(clf)

# +
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
import graphviz

X, y = df_data, df_target

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Generate DOT data from the trained classifier
dot_data = tree.export_graphviz(clf, out_file=None)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create a plot of the graph
plot = graphviz.Source(graph.to_string())

# Display the plot
plot.render("decision_tree")  # You can also use plot.view() to open it in a viewer
plot.view()

# +
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree

X, y = df_data, df_target

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Generate DOT data from the trained classifier
dot_data = tree.export_graphviz(clf)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Display the graph
graph.write_png("D:\Documents/decision_tree.png")
  # Save the graph as a PNG file
# -

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Anang/Anaconda3/Lib/site-packages/graphviz'

# +
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

X, y = df_data, df_target

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Generate DOT data from the trained classifier
dot_data = tree.export_graphviz(clf)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create a plot of the graph using graphviz
plot = graphviz.Source(graph.to_string())

# Save and view the graph
plot.render("decision_tree", format="png", cleanup=True)  # Save as PNG file and clean up temporary files
plot.view()  # Open the generated PNG file in a viewer

# +
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt

X, y = df_data, df_target

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Generate DOT data from the trained classifier
dot_data = tree.export_graphviz(clf)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Save the graph as PNG
graph.write_png("decision_tree.png")

# Display the graph using Matplotlib
image = plt.imread("decision_tree.png")
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.show()


# +
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
import io

X, y = df_data, df_target

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Generate DOT data from the trained classifier
dot_data = tree.export_graphviz(clf, out_file=None)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Generate a PNG image from the graph
image_data = graph.create_png()

# Display the image using IPython display
Image(image_data)

# +
import pydotplus
from sklearn import tree
import matplotlib.pyplot as plt
from IPython.display import Image  # Used to display images in Jupyter Notebook

# Create a decision tree classifier (clf) and fit it to your data

# Generate DOT data for the decision tree
dot_data = tree.export_graphviz(clf, out_file=None)

# Create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Save the graph as an image file (e.g., PNG)
graph_path = "decision_tree.png"
graph.write_png(graph_path)

# Display the saved image using IPython.display.Image
Image(graph_path)

# +
import pydotplus
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10)  # You can adjust the number of trees

# Fit the classifier to the data
clf.fit(df_data, df_target)

# Visualize each tree in the random forest
for i, tree_in_forest in enumerate(clf.estimators_):
    dot_data = tree.export_graphviz(tree_in_forest, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    # Save the graph as an image file (e.g., PNG)
    graph_path = f"decision_tree_{i}.png"
    graph.write_png(graph_path)

    # Load the saved image and display it using matplotlib
    image = plt.imread(graph_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Decision Tree {i}")
    plt.show()


# +
# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
# load data

# split data into X and y
X = df_data
y = df_target.values.ravel()

# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot single tree
plot_tree(model)
plt.show()
# -

plot_tree(model, num_trees=4)

plot_tree(model, num_trees=2, rankdir='LR')

# +
# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = df_data, df_target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=6, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression(max_iter=1000)
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# Supervised transformation based on gradient boosted trees
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression(max_iter=1000)
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
# -

df_target.head(90)

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# import some data to play with
X = df_data
y = df_target.values.ravel()
class_names = ['belum_sukses', 'sukses']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = svm.SVC(kernel='linear', C=1, random_state=42)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations for each cell
thresh = cm.max() / 2
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
X = df_data
y = df_target.values.ravel()
class_names = ['belum_sukses', 'sukses']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()



# +
import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

accuracy = np.trace(cm) / np.sum(cm).astype('float')
misclass = 1 - accuracy

# Define cmap before using it
cmap = plt.get_cmap('Blues')  # Or choose any other valid colormap

if cmap is None:
    cmap = plt.get_cmap('Blues')

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()

# Define target_names
target_names = ['class']

if target_names is not None:
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


thresh = cm.max() / 1.5 if normalize else cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if normalize:
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    else:
        plt.text(j, i, "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
plt.show()
