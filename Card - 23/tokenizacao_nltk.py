import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download dos recursos necessários
nltk.download('punkt')

# Carregar texto a partir de um arquivo local
with open('data/sample_text.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Tokenização em sentenças
sentencas = sent_tokenize(texto)
print("Sentenças Tokenizadas:")
for i, sent in enumerate(sentencas, 1):
    print(f"{i}: {sent}")

# Tokenização em palavras
palavras = word_tokenize(texto)
print("\nPalavras Tokenizadas:")
print(palavras)
