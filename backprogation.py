# -*- coding: utf-8 -*-
'''
The aim of this demo is to use sklearn iris dataset to implementation backprogation by python numpy
	--input: 0-80 data as train data and 80-100 as test data 
	model: two layers networks 
	--output: two calsses(binary classification)
By @winter fu
'''
import numpy as np
from sklearn import datasets

# load dataset
loaded_data = datasets.load_iris()
iris_X = loaded_data.data
iris_Y = loaded_data.target

# hyper parameters 
max_epochs = 500
lr = 0.1
# define one_hot function
def one_hot(x):
	n_classes = x.max() + 1
	n_samples = x.shape[0] 
	onehot_array = np.zeros((n_samples, n_classes))
	for i, j in enumerate(x):
		onehot_array[i][j] = 1

	return onehot_array
# split data to train and test
train_data = iris_X[:80, :]
train_label = one_hot(iris_Y[:80])
test_data = iris_X[80:100, :]
test_label = one_hot(iris_Y[80:100])

# build model
train_sample_num = len(train_data)          # the number of samples
input_dim = train_data.shape[1]   # input size
hidden_dim = 6                       # hidden layer size
output_dim = 2                        # output size

# define activation function sigmoid 
def activation_s(x):
	return 1 / (1 + np.exp(-x))

# define error loss function square 
def get_err(e):
	return 0.5*np.sum(np.square(e))

# initialization of weights and bias
w1 = 2*np.random.random((input_dim, hidden_dim)) - 1 
w2 = 2*np.random.random((hidden_dim, output_dim)) - 1

b1 = np.zeros((1, hidden_dim))
b2 = np.zeros((1,output_dim))

# ---------training--------
for step in range(max_epochs):
	# forward progation
	hidden_val = np.dot(train_data, w1) + b1
	hidden_act = activation_s(hidden_val) 
	output_val = np.dot(hidden_act, w2) + b2
	output_act = activation_s(output_val)

	# back progation
	e = train_label - output_act
	out_delta = e*output_act*(1 - output_act)
	hidden_delta = hidden_act*(1-hidden_act)*np.dot(out_delta, w2.T) 

	w2 += lr*np.dot(hidden_act.T, out_delta)
	w1 += lr*np.dot(train_data.T, hidden_delta)

	b2 += lr*np.sum(out_delta, axis=0, keepdims = True)
	b1 += lr*np.sum(hidden_delta, axis=0, keepdims = True)


# -----------------testing-------------------
number = 0

hidden_val = np.dot(test_data, w1) + b1
hidden_act = activation_s(hidden_val)
output_val = np.dot(hidden_act, w2) + b2
output_act = activation_s(output_val) 

print(test_label)
print(output_act)
for i in range(len(test_label)):
	if np.argmax(output_act[i]) == np.argmax(test_label[i]):
		number += 1

print("accuracy:" , number/len(test_label))
