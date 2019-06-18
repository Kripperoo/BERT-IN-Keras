import keras.backend as K
from keras.layers import Layer, Conv1D, Dropout, Add, Input
from keras.initializers import Ones, Zeros


def gelu(x):
	return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class LayerNorm(Layer):

	def __init__(self, eps=1e-5, **kwargs) -> None:
		self.eps = eps
		super().__init__(**kwargs)

	def build(self, input_shape):
		self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super().build(input_shape)

	def call(self, data, **kwargs):
		mean = K.mean(data, axis=-1, keepdims=True)
		std = K.mean(K.square(data-mean), axis=-1, keepdims=True)
		return self.gamma* (data-mean) / K.sqrt(std+self.eps) + self.beta


class Gelu(Layer):
	def __init__(self, accurate=False, **kwargs):
		super().__init__(**kwargs)
		self.accurate = accurate

	def call(self, data, **kwargs):
		if accurate:
			erf = K.tf.erf
			return inputs * 0.5 * (1.0 + erf(inputs / math.sqrt(2.0)))
		else:
			return gelu(data)

class PPFeedForward:
	def __init__(self, num_state, hidden_ff, droput_p=0.1, act_GeLu=False):
		self.activation = Gelu(accurate=act_GeLu)
		self.fw1 = Conv1D(hidden_ff, 1)
		self.fw2 = Conv1D(num_state, 1)
	def __call__(self, input):
		return self.fw2(self.activation(self.fw1(input)))

class ResidualConnection:
	def __init__(self, data, droput_p):
		self.layer_norm = LayerNorm()
		self.dropout = Dropout(droput_p)
	def __call__(self, x, sublayer):
		#Apply residual Connection to any sublayer
		return x + self.dropout(sublayer(self.layer_norm(x)))


