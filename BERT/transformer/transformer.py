

class Encoder:
	def __init__(self, num_state, num_head, FF_hidden_states, 
		residual_dropout, attention_dropout, mask: bool, 
		layer_id, norm_epsilon, accurate_Gelu: bool) -> None:
	self.MH_attn = MultiHeadedSelfAttention(num_state, num_head, attention_dropout, mask)
	self.sublayer1 = ResidualConnection(norm_epsilon, residual_dropout)
	self.feed_forward = PPFeedForward(num_state, FF_hidden_states, accurate_Gelu)
	self.sublayer2 = ResidualConnection(norm_epsilon, residual_dropout)
	self.drop = Dropout(residual_dropout)

def __call__(self, x, mask):
	output = self.sublayer1(x, MH_attn(x, mask))
	output = self.sublayer2(x, self.feed_forward(output))
	return self.drop(output)

def create_BERT(embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30000,
                       max_len: int = 512, trainable_pos_embedding: bool = True, num_heads: int = 12,
                       num_layers: int = 12, attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,
                       d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_attn_mask: bool = True,
                       embedding_layer_norm: bool = False, neg_inf: float = -1e9, layer_norm_epsilon: float = 1e-5,
                       accurate_gelu: bool = False) -> keras.Model:

    vocab_size += TextEncoder.SPECIAL_COUNT
    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype='int32')
    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype='int32')
    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype='int32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',
                      dtype=K.floatx()) if use_attn_mask else None

    inputs = [tokens, segment_ids, pos_ids]
    embedding_layer = Embedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                use_one_embedding_dropout, embedding_layer_norm, layer_norm_epsilon)
    x = embedding_layer(inputs)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i, layer_norm_epsilon, accurate_gelu)(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)

    return keras.Model(inputs=inputs, outputs=[x], name='Transformer')

