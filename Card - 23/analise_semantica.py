import spacy

# Carregar o modelo de linguagem em inglês
nlp = spacy.load("en_core_web_sm")

# Carregar texto a partir de um arquivo local
with open('data/sample_sentence.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Processar o texto
doc = nlp(texto)

# Análise sintática
print("Análise Sintática:")
for token in doc:
    print(f"Token: {token.text}, Dependência: {token.dep_}, Cabeça: {token.head.text}")
