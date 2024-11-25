import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np

# Camada de Transformador
class TransformerBlock(Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Modelo de Transformador
def build_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedding_layer)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Dados de exemplo
texts = ["I love machine learning", "Deep learning is fascinating", "Transformers are powerful", "GANs can generate images"]
labels = [1, 1, 1, 0]

# Pré-processamento
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = 10
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# Construção e treinamento do modelo
model = build_transformer_model(vocab_size=10000, max_len=max_len, embed_dim=32, num_heads=2, ff_dim=32)
model.summary()
model.fit(X, y, epochs=10, verbose=1)
