'''
Niharika Agrawal
Experimentation with Linear Regression Models
Intro to Machine Learning

one plane linear regression model, just an x and y axis
'''

#import necessary programs
import numpy.random # for generating a noisy data set
from sklearn import linear_model # for training a linear model
import matplotlib.pyplot # for plotting in general
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import pandas as pd

# Setting the limits and number of our first, X, variable
MIN_X = -20
MAX_X = 20
NUM_INPUTS = 75

# randomly pick numbers for x
x_one_x = numpy.random.uniform(low = MIN_X, high = MAX_X, size = (NUM_INPUTS, 1))

#create a neat column graph with the data
data_one_x = pd.DataFrame(data = x_one_x, columns = ['x'])

# generate some normally distributed noise
noise_one_x = numpy.random.normal(size = NUM_INPUTS)

#get the outputs for the randomized x data

data_one_x['y'] = 1.45 * data_one_x['x'] + 2.3 * data_one_x['x'] + noise_one_x
data_one_x.plot.scatter(x='x', y='y')

# create an empty linear model for us to put things into
model_one_x = linear_model.LinearRegression()

#reshape the data to make each individual value its own list within a list
x_one_x = data_one_x['x'].values.reshape(-1, 1)
y_one_x = data_one_x['y'].values.reshape(-1, 1)

#fit the model to the data_one_x
model_one_x.fit(X=x_one_x, y=y_one_x)

#define the parameters for printing our model results out (the intercept and coefficient)
def print_model_fit(model):
    # Print out the parameters for the best fit line
    print('Intercept: {i}  Coefficients: {c}'.format(i=model.intercept_, c=model.coef_))

print_model_fit(model_one_x)

'''
#the code for putting the data into the graph from the jupyter notebook
def plot_best_fit_line(model, x, y):
    # create the figure
    fig = matplotlib.pyplot.figure(1)
    fig.suptitle('Data and Best-Fit Line')
    matplotlib.pyplot.xlabel('x values')
    matplotlib.pyplot.ylabel('y values')

    # put the generated dataset points on the graph
    matplotlib.pyplot.scatter(x, y)

    # Now we actually want to plot the best-fit line.
    # To simulate that, we'll simply generate all the inputs on the graph and plot that.
    # predict for inputs along the graph to find the best-fit line
    X = numpy.linspace(MIN_X, MAX_X) # generates all the possible values of x
    Y = model.predict(list(zip(X)))
    matplotlib.pyplot.plot(X, Y)
'''

#new plot best fit that i'm tinkering with
def plot_best_fit_line(model, x, y):
    # create the figure
    fig = matplotlib.pyplot.figure(1)
    fig.suptitle('Data and Best-Fit Line')

    axes = fig.gca (projection='2d')
    axes.set_xlabel('x values')
    axes.set_ylabel('y values')

    # put the generated dataset points on the graph
    axes.scatter(x, y)

    # Now we actually want to plot the best-fit line by generating the inputs and graphing
    X = numpy.linspace(MIN_X, MAX_X) # generates all the possible values of x
    Y = model.predict(list(zip(X)))

    #put the line on the graph
    axes.plot(X,Y, alpha=0.5)

    #show the graph
    matplotlib.pyplot.show()

#graph for the original 50 new_x_values
plot_best_fit_line(model_one_x, x_one_x, y_one_x)
