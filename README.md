# Pesquisa Quântica - Classificação de Dígitos

Este projeto realiza uma comparação entre modelos de Machine Learning Clássico (SVM) e Quântico (QSVC) utilizando o dataset `digits` do scikit-learn. O foco da pesquisa é avaliar o desempenho e o tempo de processamento em classes específicas (dígitos 3, 5 e 8) com redução de dimensionalidade via PCA.

## Estrutura do Projeto

- `modelo_digits_classic.py`: Implementação do SVM clássico com kernel RBF.
- `modelo_digits_quantic.py`: Implementação do QSVC (Quantum Support Vector Classification) utilizando Qiskit.
- `work.py`: Script de verificação do ambiente e versões das bibliotecas.
- `requirements.txt`: Lista de dependências necessárias.
- `imagens_artigo_classico/`: (Ignorado no Git) Contém os gráficos gerados pelo modelo clássico.
- `imagens_artigo_quantico/`: (Ignorado no Git) Contém os gráficos gerados pelo modelo quântico.

## Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado. Recomenda-se o uso de um ambiente virtual.

### Instalação

1. Clone o repositório (após o upload).
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como Executar

Para rodar a verificação de ambiente:

```bash
python work.py
```

Para rodar o modelo clássico:

```bash
python modelo_digits_classic.py
```

Para rodar o modelo quântico:

```bash
python modelo_digits_quantic.py
```

_Nota: O modelo quântico utiliza simulação local e a geração da fronteira de decisão pode ser computacionalmente intensiva._

## Resultados Esperados

O projeto gera métricas de acurácia, F1-Score e visualizações das fronteiras de decisão no espaço de Hilbert (para o quântico) e no espaço 2D após PCA.
