import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Baixar recursos do NLTK
nltk.download('punkt')

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def perform_tokenization(text):
    sentencas = sent_tokenize(text)
    palavras = word_tokenize(text)
    print("Sentenças Tokenizadas:")
    for i, sent in enumerate(sentencas, 1):
        print(f"{i}: {sent}")
    print("\nPalavras Tokenizadas:")
    print(palavras)

def perform_custom_tokenization(text):
    texto_limpo = re.sub(r'[^\w\s]', '', text)
    tokens = texto_limpo.split()
    print("Tokens Personalizados:")
    print(tokens)

def perform_ner(nlp, text):
    doc = nlp(text)
    print("Entidades Nomeadas:")
    for ent in doc.ents:
        print(f"Entidade: {ent.text}, Rótulo: {ent.label_}")

def perform_syntax_analysis(nlp, text):
    doc = nlp(text)
    print("Análise Sintática:")
    for token in doc:
        print(f"Token: {token.text}, Dependência: {token.dep_}, Cabeça: {token.head.text}")

def perform_lemmatization(nlp, text):
    doc = nlp(text)
    print("Lemas das Palavras:")
    for token in doc:
        print(f"Palavra: {token.text}, Lema: {token.lemma_}")

def main():
    # Carregar modelos spaCy
    nlp = spacy.load("en_core_web_sm")
    
    # Caminhos para os arquivos de dados
    sample_text_path = 'data/sample_text.txt'
    sample_sentence_path = 'data/sample_sentence.txt'
    sample_text_lemmatization_path = 'data/sample_text_lemmatization.txt'
    sample_sentence_dependencies_path = 'data/sample_sentence_dependencies.txt'
    
    # Carregar textos
    texto = load_text(sample_text_path)
    texto_lemmatization = load_text(sample_text_lemmatization_path)
    texto_dependencies = load_text(sample_sentence_dependencies_path)
    
    # Tokenização com NLTK
    print("=== Tokenização com NLTK ===")
    perform_tokenization(texto)
    print("\n============================\n")
    
    # Tokenização personalizada com regex
    print("=== Tokenização Personalizada com Regex ===")
    perform_custom_tokenization(texto)
    print("\n===========================================\n")
    
    # Reconhecimento de Entidades Nomeadas com spaCy
    print("=== Reconhecimento de Entidades Nomeadas com spaCy ===")
    perform_ner(nlp, texto)
    print("\n======================================================\n")
    
    # Análise Sintática com spaCy
    print("=== Análise Sintática com spaCy ===")
    perform_syntax_analysis(nlp, texto_dependencies)
    print("\n===================================\n")
    
    # Lematização com spaCy
    print("=== Lematização com spaCy ===")
    perform_lemmatization(nlp, texto_lemmatization)
    print("\n=============================\n")

if __name__ == "__main__":
    main()


# Para rodar, clone repositório git clone https://github.com/wjbmattingly/freecodecamp_spacy.git
