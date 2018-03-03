from numpy import *


def compute_error_for_given_points(b, m, points):
    # Estimate how bad our line is, so that we update at
    # each timestep
    # Every time step we need to improve our model's prediction
    # We need to minimize error/minimize loss
    # We want to measure the distance from each point
    # to the line, using sum of squared errors
    # error formula: https://www.dropbox.com/s/9ap7dki5qlfpr4n/Screenshot%20from%202018-03-03%2013-47-33.png?dl=0
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
        # In the above equation, we compute the error
        # using a linear model
    # Now we have the error, calculate the average
    # of the total error
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # All of gradient descent happens here
    b_gradient = 0
    m_gradient = 0
    # The plot of error vs y-intercept vs slope is a 3 dimensional
    # bowl like curve with a local minima
    # Local minima occurs at the point that has the smallest error
    # The we calculate this point is by calculating the gradient; not
    # to be confused with the m value
    # We are talking of the slope in the direction that will give us
    # that value
    # We can think of the entire process as a bowl
    # To calculate the gradient, we'll use the partial derivative
    # w.r.t values b and m
    # Formula to calculate gradient : https://www.dropbox.com/s/xhwd7qyimdlhzzp/Screenshot%20from%202018-03-03%2014-33-53.png?dl=0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    # We have got b and m values
    # To start with it will be zero
    # We'll learn these values
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001  # hyperparameter / #turningknob
    # if our learning rate is too low, our model will be too
    # slow to converge, if it is to high, it will never converge
    # hyperparameters, we need to guess and check
    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    # how many iterations we need to run on our data set
    # why 1000?,, because our data set is small
    [b, m] = gradient_descent_runner(
        points,
        initial_b,
        initial_m,
        learning_rate,
        num_iterations
    )
    # will get the ideal value of b and m
    print(b)
    print(m)


if __name__ == '__main__':
    run()
