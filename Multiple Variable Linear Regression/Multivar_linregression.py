from numpy import *

def compute_error_for_line_given_points(b,m,points):
        totalError = 0
        for i in range(0, len(points)):
                x1 = points[i, 0]
                x2 = points[i, 1]
                x3 = points[i, 2]
                y = points[i, 3]

                totalError += (y-(m * x1 + m * x2 + m * x3 + b)) **2

        return  totalError/float(len(points))
        

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
        
	#gradient Descent Algorithm
	for i in range(num_iterations):
		b, m = step_gradient(b,m, array(points), learning_rate)

	return ([b,m])

def step_gradient(b_current,m_current, points, learningRate):
        
	b_gradient = 0
	m_gradient = 0

	N = float(len(points))
        
	for i in range(0, len(points)):
		x1 = points[i, 0]
		x2 = points[i, 1]
		x3 = points[i, 2]
		y = points[i, 3]


		#direction with respect to b and m
		#computing partial derivative of our error function

		b_gradient += -(2/N) * (y - ((m_current * x1 + m_current * x2 + m_current * x3) + b_current))
		m_gradient += -(2/N) * (x1+x2+x3) * (y - ((m_current * x1 + m_current * x2 + m_current * x3) + b_current))

    #updating the b and m values using the partial derivatives
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]



def run():
	#1
	points = genfromtxt('data.csv',delimiter=',')


	#2
	#define hyperparameters
	learning_rate = 0.0001
	#y=mx+b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000


	#3 training our model
	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
	print ("Running...")
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


run()