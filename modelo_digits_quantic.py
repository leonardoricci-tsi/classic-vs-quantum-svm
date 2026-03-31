import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator

# --- CONFIGURAÇÃO DE DIRETÓRIO ---
output_dir = "imagens_artigo_quantico"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Preparação dos Dados (Classes 3, 5, 8)
digits = load_digits()
mask = (digits.target == 3) | (digits.target == 5) | (digits.target == 8)
X, y = digits.data[mask], digits.target[mask]

# Normalização e PCA 2D
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 2. Configuração Quântica
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
q_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=q_kernel)

print(f"Iniciando treinamento Quântico para as classes {np.unique(y)}...")
start_time = time.time()
qsvc.fit(X_train, y_train)
end_time = time.time()

process_time_q = end_time - start_time

# 4. Avaliação e Métricas
y_pred_q = qsvc.predict(X_test)
acc_q = accuracy_score(y_test, y_pred_q)
matrix_train = q_kernel.evaluate(x_vec=X_train) 

print(f"\n--- RESULTADOS QUÂNTICOS (3, 5, 8 | PCA 2D) ---")
print(f"Acurácia Quântica: {acc_q:.4f}")
print(f"Tempo de Processamento Quântico: {process_time_q:.4f} segundos")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_q, target_names=['Dígito 3', 'Dígito 5', 'Dígito 8']))


# Imagem 1: Projeção do Espaço de Hilbert
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_train[:15, :15], annot=True, cmap='magma', cbar=True)
plt.title(f"Espaço de Hilbert (Matriz de Fidelidade)\nTempo: {process_time_q:.2f}s")
plt.savefig(os.path.join(output_dir, "01_hilbert_matrix_quantum.png"), dpi=300, bbox_inches='tight')
plt.close()

classes = ['Dígito 3', 'Dígito 5', 'Dígito 8']
f1_quantum = f1_score(y_test, y_pred_q, average=None)

plt.figure(figsize=(12, 6))
plt.plot(classes, f1_quantum, marker='s', linestyle='--', linewidth=2, label='QSVC Quântico', color='#f3750c')

plt.title("Performance Quântica por Classe (F1-Score)")
plt.ylabel("F1-Score")
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

for i, txt in enumerate(f1_quantum):
    plt.annotate(f"{txt:.2f}", (classes[i], f1_quantum[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#f3750c')

plt.savefig(os.path.join(output_dir, "02_performance_f1_quantum.png"), dpi=300, bbox_inches='tight')
plt.close()

def save_quantum_decision_boundary(X, y, model, title, folder):
    print("\nGerando fronteira de decisão quântica (isso pode demorar alguns minutos)...")
    h = .1 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis', s=45)
    
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filepath = os.path.join(folder, "03_fronteira_decisao_quantica.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Imagem da fronteira quântica salva em: {filepath}")
    plt.show()

save_quantum_decision_boundary(X_pca, y, qsvc, 
                              f"QSVC Quântico: Dígitos 3, 5 e 8 (PCA 2D)\nAcurácia: {acc_q:.2f}", 
                              output_dir)

print(f"\nTodas as imagens quânticas salvas em: {os.path.abspath(output_dir)}")