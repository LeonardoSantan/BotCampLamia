import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Parte 1: GAN para Geração de Imagens
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(28*28*1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_gan(gan, generator, discriminator, epochs=5000, batch_size=128):
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    
    for epoch in range(epochs):
        # Dados reais
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        # Dados falsos
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_imgs = generator.predict(noise)
        
        # Treinamento do discriminador
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Treinamento do gerador
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
            plot_generated_images(generator)

def plot_generated_images(generator, examples=5, dim=(1,5), figsize=(10,2)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Parte 2: Transformador para Geração de Descrições
class TransformerBlock(Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(max_len,))
    embedding_layer = layers.Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedding_layer)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Dados de exemplo para descrições
descriptions = ["A handwritten digit.", "A clear image of a digit.", "Another handwritten number.", "An example of a generated digit."]
labels = [1, 1, 0, 0]  # Exemplos simplificados

# Pré-processamento
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(descriptions)
sequences = tokenizer.texts_to_sequences(descriptions)
max_len = 10
X_desc = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
y_desc = np.array(labels)

# Construção e treinamento do transformador
transformer_model = build_transformer_model(vocab_size=10000, max_len=max_len, embed_dim=32, num_heads=2, ff_dim=32)
transformer_model.summary()
transformer_model.fit(X_desc, y_desc, epochs=10, verbose=1)

# Projeto Final: Geração de Imagens com Descrições
def generate_image_with_description(generator, transformer_model, tokenizer):
    # Gerar imagem
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)
    generated_image = generated_image.reshape(28, 28)
    
    # Mostrar imagem
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Gerar descrição
    sample_desc = "A handwritten digit."  # Placeholder
    sequence = tokenizer.texts_to_sequences([sample_desc])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=10)
    prediction = transformer_model.predict(sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    description = "Generated digit appears clear." if predicted_label == 1 else "Generated digit is unclear."
    
    print(f"Descrição Gerada: {description}")

# Execução do Projeto Final
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
train_gan(gan, generator, discriminator, epochs=5000, batch_size=128)

generate_image_with_description(generator, transformer_model, tokenizer)
