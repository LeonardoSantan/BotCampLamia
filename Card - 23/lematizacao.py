import spacy

# Carregar o modelo de linguagem em inglês
nlp = spacy.load("en_core_web_sm")

# Carregar texto a partir de um arquivo local
with open('data/sample_text_lemmatization.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Processar o texto
doc = nlp(texto)

# Lematização
print("Lemas das Palavras:")
for token in doc:
    print(f"Palavra: {token.text}, Lema: {token.lemma_}")
