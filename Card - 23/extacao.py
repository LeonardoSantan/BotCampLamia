import spacy
from spacy import displacy

# Carregar o modelo de linguagem em inglês
nlp = spacy.load("en_core_web_sm")

# Carregar texto a partir de um arquivo local
with open('data/sample_sentence_dependencies.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Processar o texto
doc = nlp(texto)

# Extração de dependências
print("Dependências das Palavras:")
for token in doc:
    print(f"Palavra: {token.text}, Dependência: {token.dep_}, Cabeça: {token.head.text}")

# Visualização gráfica das dependências (requer ambiente com interface gráfica)
# displacy.serve(doc, style="dep")
