# Binary Knapsack Problem

Este projeto implementa vários algoritmos para resolver o problema da mochila 0-1 (0/1 Knapsack Problem), incluindo heurísticas aproximadas e métodos exatos. Suporta avaliação em conjuntos de instâncias **low-dimensional** (pequenas) e **large-scale** (grandes), com medição de tempo e uso de memória.

## 📦 Estrutura do Projeto

- **main.py**: ponto de entrada pela linha de comando
- **runner.py**: orquestra a execução, multiprocessos e timeout
- **algorithms.py**: implementação dos algoritmos (2-Approx, FPTAS, Branch and Bound, Backtracking)
- **io_utils.py**: leitura e escrita de arquivos de instâncias/resultados
- **utils.py**: utilitários gerais (timer, impressão de cabeçalhos)

## ✅ Pré-requisitos

- Python 3.8 ou superior
- Pacote `psutil` (usado para medir uso de memória)

## ✅ Instalação de dependências

Você pode instalar `psutil` globalmente (sem usar ambiente virtual):

```bash
pip install psutil
```

## ✅ Organização dos dados

O projeto espera duas pastas principais com as instâncias:

```
./low-dimensional
./large_scale
```

### Low-dimensional

Cada arquivo deve conter:

```
n capacity
value weight
value weight
...
```

### Large-scale

Dois arquivos por instância:

- **knapPI_<index>_<n_items>_<capacity>_<other>_items.csv**
  - Cabeçalho: `price,weight`
  - Itens com valores e pesos.
  
- **knapPI_<index>_<n_items>_<capacity>_<other>_info.csv**
  - Contém a capacidade (`c`) e/ou o ótimo conhecido (`z`).

## ✅ Como executar

### Avaliar todas as instâncias

Para rodar **todos os conjuntos** (low e large):

```bash
python main.py
```

Para rodar **apenas low-dimensional**:

```bash
python main.py low
```

Para rodar **apenas large-scale**:

```bash
python main.py large
```

### Rodar uma instância específica

Use a flag `--single` com o nome exato da instância:

```bash
python main.py low --single f3_l-d_kp_4_20
```

```bash
python main.py large --single knapPI_14_200_1000_1
```

## ✅ Resultados

Ao fim da execução, o script salva os resultados em CSV:

```
results_low-dimensional.csv
results_large_scale.csv
```

Cada linha contém:

- Nome da instância
- Número de itens
- Capacidade
- Algoritmo
- Valor obtido
- Ótimo conhecido (se existir)
- Fator de aproximação
- Tempo de execução
- Uso de memória

## ✅ Uso de ambiente virtual (opcional)

Embora **não seja obrigatório**, recomenda-se usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
.
env\Scripts ctivate       # Windows

pip install -r requirements.txt
```

**Exemplo de requirements.txt:**

```
psutil
```
