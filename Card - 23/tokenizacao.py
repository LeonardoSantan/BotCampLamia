import re

def custom_tokenizer(texto):
    # Remover pontuação utilizando regex
    texto_limpo = re.sub(r'[^\w\s]', '', texto)
    # Dividir o texto em tokens com base em espaços
    tokens = texto_limpo.split()
    return tokens

# Carregar texto a partir de um arquivo local
with open('data/sample_text.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Tokenização personalizada
tokens = custom_tokenizer(texto)
print("Tokens Personalizados:")
print(tokens)
