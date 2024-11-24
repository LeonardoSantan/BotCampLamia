import spacy

# Carregar o modelo de linguagem em inglês
nlp = spacy.load("en_core_web_sm")

# Carregar texto a partir de um arquivo local
with open('data/sample_text.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Processar o texto
doc = nlp(texto)

# Extração de entidades nomeadas
print("Entidades Nomeadas:")
for ent in doc.ents:
    print(f"Entidade: {ent.text}, Rótulo: {ent.label_}")
