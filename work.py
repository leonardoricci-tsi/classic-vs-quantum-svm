import qiskit
import qiskit_aer
from qiskit_machine_learning.algorithms import QSVC

print(f"Qiskit version: {qiskit.__version__}")
print(f"Simulador Aer pronto: {qiskit_aer.AerSimulator()}")
print("Ambiente configurado com sucesso para o artigo!")