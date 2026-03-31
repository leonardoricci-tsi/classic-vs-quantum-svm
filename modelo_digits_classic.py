import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

output_dir = "imagens_artigo_classico"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Carregamento e Filtragem (Classes: 3, 5 e 8)
digits = load_digits()
X_all, y_all = digits.data, digits.target
mask = (y_all == 3) | (y_all == 5) | (y_all == 8)
X, y = X_all[mask], y_all[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA delimitado para 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

model = SVC(kernel='rbf', C=1.0)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

process_time = end_time - start_time

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--- RESULTADOS DIGITS (CLASSES 3, 5, 8 | PCA 2D) ---")
print(f"Acurácia Clássica: {acc:.4f}")
print(f"Tempo de Processamento: {process_time:.6f} segundos")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Dígito 3', 'Dígito 5', 'Dígito 8']))

def save_decision_boundary(X, y, model, title, folder):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis', s=45)
    
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Salvando em alta resolução
    filepath = os.path.join(folder, "fronteira_decisao_classica.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nImagem da fronteira salva em: {filepath}")
    plt.show()

save_decision_boundary(X_pca, y, model, 
                       f"SVM Clássico: Dígitos 3, 5 e 8 (PCA 2D)\nAcurácia: {acc:.2f}", 
                       output_dir)