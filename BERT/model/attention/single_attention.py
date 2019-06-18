import keras.backend as K
from keras.layers import Dropout

def scaled_dot_product_attention(query, key, value, mask=None, dropout: float):
	scores = K.batch_dot(query, key) / K.sqrt(K.cast(value.shape[-1], K.floatx()))

	if mask is not None:
		scores = mask * scores + (1.0 - mask) * 1e-9

	scores = K.softmax(scores)
	scores = Dropout(dropout)(scores)
	p_attention = K.batch_dot(scores, value)
	return p_attention





