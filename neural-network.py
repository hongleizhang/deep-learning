#encoding:utf-8

'''
@author:zhanghonglei
@email:hongleizhang@bjtu.edu.cn
@date:2017-08-21
'''

import numpy as np
import pandas as pd
import scipy as sp

class neuralNetwork(object):
	"""docstring for neuralNetwork
	just for three layers network
	just for better understanding the process of updating weights
	refer to the book make your own neural network
	"""
	def __init__(self, input_node_num,hidden_node_num,output_node_num,learning_rate):
		super(neuralNetwork, self).__init__()

		#initialize the number of nodes in the input,hidden,output layer
		self.input_node_num = input_node_num
		self.hidden_node_num = hidden_node_num
		self.output_node_num = output_node_num

		#initialize the weights of the whole network
		self.W_ih=np.random.normal(0.0,pow(self.hidden_node_num,-0.5),\
			(self.hidden_node_num,self.input_node_num))
		self.W_ho=np.random.normal(0.0,pow(self.output_node_num,-0.5),\
			(self.output_node_num,self.hidden_node_num))

		#set learning rate
		self.learning_rate=learning_rate
		#define the activation function,here is the sigmoid function
		self.activation_func=lambda x: sp.special.expit(x)

	def train(self,input_list,target_list):
		#convert the input data to 2d array form
		inputs=np.array(input_list,ndmin=2).T
		#convert the target data to 2d array form
		targets=np.array(target_list,ndmin=2).T
		
		#=======forward propagation========#
		#matrix multiplication for inputs of hidden layer
		hidden_inputs=np.dot(self.W_ih,inputs)
		#calculate the signals after non-linear transform
		hidden_outputs=self.activation_func(hidden_inputs)

		#matrix multiplication for inputs of output layer
		output_inputs=np.dot(self.W_ho,hidden_outputs)
		#calculate the signals after non-linear transform
		output_outputs=self.activation_func(output_inputs)

		#=======back propagation========#
		#output layer errors
		output_errors=targets-output_outputs
		#hidden layer errors
		hidden_errors=np.dot(self.W_ho.T,output_errors)

		#update weights W_ho by gradient descent
		self.W_ho+=self.learning_rate*np.dot((output_errors*output_outputs*\
			(1.0-output_outputs)),np.transpose(hidden_outputs))
		#update weights W_ih by gradient descent
		self.W_ih+=self.learning_rate*np.dot((hidden_errors*hidden_outputs*\
			(1.0-hidden_outputs)),np.transpose(inputs))
		pass

	def query(self,input_list):

		#convert the input data to 2d array form
		inputs=np.array(input_list,ndmin=2).T

		#matrix multiplication for inputs of hidden layer
		hidden_inputs=np.dot(self.W_ih,inputs)
		#calculate the signals after non-linear transform
		hidden_outputs=self.activation_func(hidden_inputs)

		#matrix multiplication for inputs of output layer
		output_inputs=np.dot(self.W_ho,hidden_outputs)
		#calculate the signals after non-linear transform
		output_outputs=self.activation_func(output_inputs)

		return output_outputs
