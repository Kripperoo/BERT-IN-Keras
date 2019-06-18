from single_attention import scaled_dot_product_attention
import keras.backend as K
from keras.layers import Dropout 
from keras.layers import Layers

def mutihead_attention(data, mask, num_heads: int, num_state: int, dropout : float):
	d_k = shapes[-1] // num_heads
	query, key, value = data[:, :, :num_state], data[:, :, num_state:2*num_state], data[:, :, -num_state:]
	query, key, value= [K.reshape(ele, [-1, ele.shape[1], num_heads, d_k]) for ele in [query,key, value]]

	query, value = [K.permute_dimensions(ele, [0,2,3,1]) for ele in [query, value]]
	key = K.permute_dimensions(key [0,2,1,3])

	attention_scores = scaled_dot_product_attention(query, key, value, mask, dropout)
	attention_scores = K.permute_dimensions(attention_scores, [0,2,1,3])
	merged_scores = K.reshape(attention_scores, [attention_scores.shape[0], attention_scores.shape[1], attention_scores.shape[2]*attention_scores.shape[3]])

	return merged_scores


class MultiHeadedAtten(Layer):
	def __init__(self, num_heads, num_state, dropout, mask, **kwargs) -> None:
		super().__init__(**kwargs)

		self.num_heads = num_heads
		self.num_state = num_state
		self.dropout = dropout
		self.mask = mask

	def call(self, data, **kwargs):
		x = data[0] if self.mask else data
		att_mask = data[1] if self.mask else None
		return mutihead_attention(x, mask, self.num_heads, self.num_state, self.dropout)





