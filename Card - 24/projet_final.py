import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

import shap

import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    # Carregar o conjunto de dados
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC AUC': auc
        })
    return pd.DataFrame(results)

def plot_performance(results):
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='ROC AUC', data=results)
    plt.title('Comparação de Modelos de Machine Learning - ROC AUC')
    plt.xticks(rotation=45)
    plt.show()

def shap_analysis(model, X_train, feature_names):
    # Explicador SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    # Resumo das importâncias
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, plot_type="bar")
    
    # Gráfico de dependência para a feature mais importante
    shap.dependence_plot(0, shap_values.values, X_train, feature_names=feature_names)

def main():
    # Carregar e pré-processar os dados
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Treinar os modelos
    models = train_models(X_train, y_train)
    
    # Avaliar os modelos
    results = evaluate_models(models, X_test, y_test)
    print("Desempenho dos Modelos:")
    print(results)
    
    # Plotar o desempenho
    plot_performance(results)
    
    # Selecionar o melhor modelo baseado no ROC AUC
    best_model_name = results.loc[results['ROC AUC'].idxmax()]['Model']
    best_model = models[best_model_name]
    print(f"\nMelhor Modelo: {best_model_name}")
    
    # Análise SHAP no melhor modelo
    print("\nAnálise SHAP:")
    shap_analysis(best_model, X_train, feature_names)
    
    # Relatório de classificação do melhor modelo
    y_pred = best_model.predict(X_test)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {best_model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

if __name__ == "__main__":
    main()
