'''
Niharika Agrawal
Experimentation with Linear Regression Models
Intro to Machine Learning

3d linear regression model, data is still along just one plane without a linear regression line
'''

#import necessary programs
import numpy.random # for generating a noisy data set
from sklearn import linear_model # for training a linear model
import matplotlib.pyplot # for plotting in general
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import pandas as pd

# Setting the limits and number of our first, X, variable
MIN_X = -10
MAX_X = 10
NUM_INPUTS = 50

# generate some normally distributed noise
noise_two_x = numpy.random.normal(size=NUM_INPUTS)

# randomly pick pairs of numbers for x
x1_two_x = numpy.random.uniform(low=MIN_X, high=MAX_X, size=NUM_INPUTS)
x2_two_x = numpy.random.uniform(low=MIN_X, high=MAX_X, size=NUM_INPUTS)

y_two_x = 0.5 * x1_two_x - 2.7 * x2_two_x - 2 + noise_two_x

data_two_x = pd.DataFrame(data=x1_two_x, columns = ['x1'])

data_two_x['x2'] = x2_two_x
data_two_x['y'] = y_two_x

data_two_x.head()

# use scikit-learn's linear regression model and fit to our data
model_two_x = linear_model.LinearRegression()
model_two_x.fit(data_two_x[['x1', 'x2']], data_two_x['y'])

#print model fit program
def print_model_fit(model):
    # Print out the parameters for the best fit line
    print('Intercept: {i}  Coefficients: {c}'.format(i=model.intercept_, c=model.coef_))

# Print out the parameters for the best fit plane
print_model_fit(model_two_x)

## Now create a function that can plot in 3D
def plot_3d(model, x1, x2, y):
    # 3D Plot
    # create the figure
    fig = matplotlib.pyplot.figure(1)
    fig.suptitle('3D Data and Best-Fit Plane')

    # get the current axes, and tell them to do a 3D projection
    axes = fig.gca(projection='3d')
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    axes.set_zlabel('y')


    # put the generated points on the graph
    axes.scatter(x1, x2, y)

    # predict for input points across the graph to find the best-fit plane
    # and arrange them into a grid for matplotlib
    X1 = X2 = numpy.arange(MIN_X, MAX_X, 0.05)
    X1, X2 = numpy.meshgrid(X1, X2)
    Y = numpy.array(model.predict(list(zip(X1.flatten(), X2.flatten())))).reshape(X1.shape)

    # put the predicted plane on the graph
    axes.plot_surface(X1, X2, Y, alpha=0.1)

    # show the plots
    matplotlib.pyplot.show()

#Now let's use the function
plot_3d(model_two_x, x1_two_x, x2_two_x, y_two_x)
