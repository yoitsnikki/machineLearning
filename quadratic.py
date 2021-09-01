'''
Nikki Agrawal
Experimentation with Linear Regression Models
Intro to Machine Learning

Quadratic Linear Regression

'''
'''
Sources for Editing and Working on a Better Quadratic code

https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2
https://stackoverflow.com/questions/33710829/linear-regression-with-quadratic-terms
'''


#import necessary programs
import numpy.random # for generating a noisy data set
import matplotlib.pyplot # for plotting in general
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import pandas as pd

'''
#from sklearn import linear_model # for training a linear model

#from sklearn import preprocessing
#from sklearn import pipeline

#from sklearn import * (imports everything)
#from sklearn.preprocessing import PolynomialFeatures  #for a quadratic model I think
#from sklearn.pipeline import make_pipeline #also for a polynomial/quadratic model
#from sklearn.linear_model import LinearRegression

it seems like all of this is specific packages, and so I can't just import something from sklearn
I have to import everything from a specific package of sklearn using the *
'''

from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.pipeline import *

# Setting the limits and number of our first, X, variable
MIN_X = -20
MAX_X = 20
NUM_INPUTS = 75

# randomly pick numbers for x
x_quadratic = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 1))
data_quadratic = pd.DataFrame(data=x_quadratic, columns=['x'])

# Let's create some noise to make our data a little bit more spread out.
noise_quadratic = numpy.random.normal(size=NUM_INPUTS)

# y = 0.7x^2 - 0.4x + 1.5 (the original formula, probably not in here right now)
data_quadratic['y'] = 0.7 * data_quadratic['x']**3 - 0.4 * data_quadratic['x']**2 + 1.5 * data_quadratic['x'] + 2.1 + noise_quadratic

#multiple other y options for me to play around with including the original, just copy paste the one I am using
#also replaced the original unwieldy coding with something a bit cleaner, using ** for the degrees

'''
data_quadratic['y'] = 0.7 * data_quadratic['x']**2 - 0.4 * data_quadratic['x'] + 1.5 + noise_quadratic
data_quadratic['y'] = 0.7 * data_quadratic['x']**3 - 0.4 * data_quadratic['x']**2 + 1.5 * data_quadratic['x'] + 2.1 + noise_quadratic
data_quadratic['y'] = 0.7 * data_quadratic['x']**4 - 0.4 * data_quadratic['x']**3 + 1.5 * data_quadratic['x']**2 + 2.1 * data_quadratic['x'] + 3.4 + noise_quadratic
'''

data_quadratic.plot.scatter(x='x', y='y')

# get a 1D array of the input data
x_quadratic = data_quadratic['x'].values.reshape(-1, 1)
y_quadratic = data_quadratic['y'].values.reshape(-1, 1)

# adjust the number in PolynomialFeatures() to change how many curves you want, if your data is a
#polynomial of degree three then put in three, if it is of degree 2 then put in degree 2,
#otherwise it will default to a straight line, as it is right now with no input
model_quadratic = make_pipeline(PolynomialFeatures(), LinearRegression())
#linear_model.QuadraticRegression()
model_quadratic.fit(x_quadratic, y_quadratic)

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

# show results
plot_best_fit_line(model_quadratic, x_quadratic, y_quadratic)
