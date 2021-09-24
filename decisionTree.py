'''
Nikki Agrawal
Wes Chao
Intro to Machine Learning
Decision Trees
'''

import sklearn
from sklearn.datasets import load_iris # the iris dataset is included in scikit-learn
from sklearn import tree # for fitting our model to a tree
from sklearn import linear_model # for fitting our model to a regression

# these are all needed for the particular visualization we're doing
from six import StringIO
import pydot
import os.path

import pandas #visualize data easier

# to display graphs in this notebook
import matplotlib.pyplot

# force numpy not to use scientific notation, to make it easier to read the numbers the program prints out
import numpy
numpy.set_printoptions(suppress=True)

#things i need for random forest experimentation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#the iris dataset
iris = load_iris()
iris.keys()

#view the iris dataset better
iris_df = pandas.DataFrame(iris.data)
iris_df.columns = iris.feature_names
iris_df['target'] = [iris.target_names[target] for target in iris.target]
iris_df.head()
iris_df.describe()

'''The actual charting of the Iris data'''
# Plot two of the features, 4 columns overall
x1_feature = 0
x2_feature = 3

'''
feature 0: sepal length in cm
feature 1: sepal width in cm
feature 2: petal length in cm
feature 3: petal width in cm
'''

x1 = iris.data[:,x1_feature]
x2 = iris.data[:,x2_feature]
iris_inputs = iris.data[:,[x1_feature,x2_feature]]

# The data are in order by type.
start_type_one = list(iris.target).index(0)
start_type_two = list(iris.target).index(2)

# DECISION TREE GRAPH
fig = matplotlib.pyplot.figure()
fig.suptitle('Two Features of the Iris Data Set')
matplotlib.pyplot.xlabel(iris.feature_names[x1_feature])
matplotlib.pyplot.ylabel(iris.feature_names[x2_feature])

# put the input data on the graph, with different colors and shapes for each type
scatter_0 = matplotlib.pyplot.scatter(x1[:start_type_one], x2[:start_type_one],
                                      c="red", marker="o", label=iris.target_names[0])
scatter_1 = matplotlib.pyplot.scatter(x1[start_type_one:start_type_two], x2[start_type_one:start_type_two],
                                      c="blue", marker="^", label=iris.target_names[1])
scatter_2 = matplotlib.pyplot.scatter(x1[start_type_two:], x2[start_type_two:],
                                      c="yellow", marker="*", label=iris.target_names[2])

# Train the model
model = tree.DecisionTreeClassifier()
model.fit(iris.data, iris.target)

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,
                     filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
# graph.write_pdf(os.path.expanduser("~/Desktop/iris_decision_tree.pdf"))

# Use the first input from each class
inputs = [iris.data[0], iris.data[start_type_one], iris.data[start_type_two]]

print('Class predictions: {0}'.format(model.predict(inputs))) # guess which class
print('Probabilities:\n{0}'.format(model.predict_proba(inputs))) # give probability of each class


#LOGISTIC REGRESSION graph

model = linear_model.LogisticRegression()
model.fit(iris_inputs, iris.target)

print('Intercept: {0}  Coefficients: {1}'.format(model.intercept_, model.coef_))

#plot the graph
matplotlib.pyplot.legend(handles=[scatter_0, scatter_1, scatter_2]) #graph legend
matplotlib.pyplot.show() #show the graph

'''
# RANDOM FOREST CLASSIFIER WORK

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)


X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

#decision tree classifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print("decision tree classifier: ")
scores.mean()

#random forest classifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print("random forest classifier: ")
scores.mean()

#extra trees classifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print("extra trees classifier: ")
scores.mean()
'''
