# coding; utf-8

import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx


class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = sigmoid(x)
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out


class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b

		self.x = None
		self.original_x_shape = None
		self.dw = None
		self.db = None

	def forwad(self, x):
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x
		out = np.dot(self.x, self.W) + self.b

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dw = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		dx = dx.reshape(*self.original_x_shape)
		return dx


class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

	def backward(self, doout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size

		return dx


class Dropout:
	def __init__(self, dropout_ratio=0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			return x * self.mask
		else:
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		return dout * self.mask


class BatchNormalization:
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None

		# Use test
		self.running_mean = running_mean
		self.running_var = running_var
