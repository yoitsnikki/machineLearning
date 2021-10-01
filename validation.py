'''
Nikki Agrawal
Validation on my Linear Regression Models
Intro to Machine Learning

Using the Quadratic Linear Regression

'''

'''
Sources

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
https://towardsdatascience.com/r-squared-recipe-5814995fa39a
https://www.kite.com/python/answers/how-to-take-root-mean-square-error-(rmse)-in-python
https://www.kite.com/python/docs/math.sqrt

'''

#import necessary programs
import numpy.random # for generating a noisy data set
import matplotlib.pyplot # for plotting in general
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import pandas as pd #for graphing data in tables
import math #for complicated math operations

#all my sklearn packages
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.metrics import *

'''

#from sklearn import * (imports everything)

it seems like all of this is specific packages, and so I can't just import something from sklearn
I have to import everything from a specific package of sklearn using the *
'''

# Setting the limits and number of our first, X, variable
MIN_X = -20
MAX_X = 20
NUM_INPUTS = 50

# randomly pick numbers for x
x_quadratic = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 1))
data_quadratic = pd.DataFrame(data=x_quadratic, columns=['x'])

# Let's create some noise to make our data a little bit more spread out.
noise_quadratic = numpy.random.normal(size=NUM_INPUTS)

# y = 0.7x^2 - 0.4x + 1.5 (the original formula, probably not in here right now)
data_quadratic['y'] = 0.7 * data_quadratic['x']**2 - 0.4 * data_quadratic['x'] + 1.5 + noise_quadratic
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

#find the predicted results based on my validation models, of 2 types, in the code above
'''
R2 - 1 - sum((y predicted - true y)^2)/sum()(average y - true y)^2)
RMSE - sum((predicted y - true y)^2)/total datapoints
'''

#y predicted value
y_predicted = model_quadratic.predict(list(zip(numpy.linspace(MIN_X, MAX_X))))

#R2 VALIDATION
R2 = r2_score(y_quadratic, y_predicted)
print("R2 Validation: ")
print(R2)

#RMSE Validation
rmse = math.sqrt(mean_squared_error(y_quadratic, y_predicted))
print ("RMSE Validation: ")
print(rmse)

# show results
plot_best_fit_line(model_quadratic, x_quadratic, y_quadratic)
