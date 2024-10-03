# %% [code]
# %% [code]
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def casual_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)
    return tf.tile(mask, mult)


np.transpose(casual_mask(1, 10, 10, dtype=tf.int32)[0])


@keras.utils.register_keras_serializable()
class Encoder_block(layers.Layer):
    def __init__(self, num_head, key_dim, ff_dim, dropout=0.1, **kwargs):
        super(Encoder_block, self).__init__(**kwargs)
        self.num_head = num_head
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.drop = layers.Dropout(self.dropout)
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.ff = layers.Dense(self.ff_dim, activation=layers.LeakyReLU(negative_slope=0.5))
        self.attn = layers.MultiHeadAttention(self.num_head, self.key_dim)

    def call(self, inputs):
        ln_ip = self.ln(inputs)
        attn_op, attn_scores = self.attn(ln_ip, ln_ip, return_attention_scores=True)
        attn_op = inputs + attn_op
        ln2 = self.ln(attn_op)
        ff = self.ff(ln2)
        ff = self.drop(ff)
        ff1 = self.ff(ff)
        ff1 = self.drop(ff1)
        ff_op = ff1 + attn_op
        ln_op = self.ln(ff_op)
        return ln_op

    def get_config(self):
        config = super(Encoder_block, self).get_config().copy()
        config.update({
            'num_head': self.num_head,
            'key_dim': self.key_dim,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable()
class Transformer_block(layers.Layer):
    def __init__(self, num_head, key_dim, output_shape, ff_dim, ff2_dim, dropout=0.1, **kwargs):
        super(Transformer_block, self).__init__(**kwargs)
        self.num_head = num_head
        self.key_dim = key_dim
        self.output_shape = output_shape
        self.ff_dim = ff_dim
        self.ff2_dim = ff2_dim
        self.dropout = dropout
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = layers.Dense(self.ff_dim, activation=tf.keras.layers.LeakyReLU(negative_slope=0.5))
        self.ff2 = layers.Dense(self.ff2_dim, activation=tf.keras.layers.LeakyReLU(negative_slope=0.5))
        self.ff3 = layers.Dense(self.output_shape)
        self.Drop = layers.Dropout(self.dropout)
        self.attn = layers.MultiHeadAttention(self.num_head, self.key_dim)
        self.cross = layers.MultiHeadAttention(self.num_head, self.key_dim)

    def call(self, inputs, encoder_op):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        casual = casual_mask(batch_size, seq_len, seq_len, tf.bool)
        ln_ip = self.ln(inputs)
        attn_op, attn_scores = self.attn(ln_ip, ln_ip, attention_mask=casual, return_attention_scores=True)
        attn_op = inputs + attn_op
        attn_op = self.Drop(attn_op)
        #         out1=self.ln(inputs+attn_op)
        ln2_ip = self.ln2(attn_op)
        cross = self.cross(ln2_ip, encoder_op, encoder_op)
        cross = cross + attn_op
        ln3_ip = self.ln2(cross)
        ff1 = self.ff(ln3_ip)
        #         ff1 = inputs+ff1
        ff2 = self.ff2(ff1)
        #         ff2=ff2+inputs
        ff3 = self.ff3(ff2)
        ff3 = cross + ff3
        #         ff3=ff3*10
        ff_op = self.Drop(ff3)
        return (ff_op, attn_scores)

    def get_config(self):
        config = super(Transformer_block, self).get_config().copy()
        config.update({
            'num_head': self.num_head,
            'key_dim': self.key_dim,
            'output_shape': self.output_shape,
            'ff_dim': self.ff_dim,
            'ff2_dim': self.ff2_dim,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable()
class Positional_encoding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(Positional_encoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)

    def build(self, input_shape):
        # Initialize the positional encodings once for the given `maxlen` and `embed_dim`
        self.positional_encodings = self.compute_positional_encoding(self.maxlen, self.embed_dim)
        super(Positional_encoding, self).build(input_shape)

    def compute_positional_encoding(self, maxlen, embed_dim):
        position = np.arange(maxlen)[:, np.newaxis]  # Shape (maxlen, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))  # Shape (embed_dim/2,)

        pos_enc = np.zeros((maxlen, embed_dim))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)

        return tf.constant(pos_enc, dtype=tf.float32)

    def call(self, x):
        x = self.token_emb(x)
        positional_encodings = tf.slice(self.positional_encodings, [0, 0], [tf.shape(x)[1], -1])
        x += positional_encodings

        return x

    def get_config(self):
        config = super(Positional_encoding, self).get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)