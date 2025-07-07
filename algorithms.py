import time
import psutil
import os

"""
Algoritmos para a mochila 0-1:
1. 2-Aproximado
2. FPTAS (Fully Polynomial-Time Approximation Scheme)
3. Branch and Bound
4. Backtracking

Todos os algoritmos retornam:
- Valor total
- Vetor de solução (0 ou 1 para cada item)
- Tempo de execução
- Uso de memória em MB
"""

def new_2approx(values, weights, capacity):
    """
    2-Aproximado:
    - Ordena os itens por valor/peso em ordem decrescente.
    - Inclui o máximo possível dentro da capacidade.
    - Aproximação com fator no máximo 2.
    """
    n = len(values)
    if n == 0 or capacity <= 0:
        return 0, [0] * n, 0, 0

    # Ordena itens por razão valor/peso (maior primeiro)
    items = sorted(
        range(n),
        key=lambda i: (values[i] / weights[i]) if weights[i] != 0 else float('inf'),
        reverse=True
    )

    total_value = 0
    solution = [0] * n
    remaining_capacity = capacity

    start_time = time.time()
    for i in items:
        if weights[i] <= remaining_capacity:
            solution[i] = 1
            total_value += values[i]
            remaining_capacity -= weights[i]

    # Medição de memória
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2

    return total_value, solution, time.time() - start_time, memory


def fptas(values, weights, capacity, epsilon):
    """
    FPTAS:
    - Escala os valores para reduzir o tamanho da tabela de programação dinâmica.
    - Usa aproximação controlada por epsilon.
    
    Parâmetro:
    - epsilon: fator de precisão.
    """
    n = len(values)
    start_time = time.time()

    # Determina o maior valor para definir a escala
    vmax = max(values)
    scale = epsilon * vmax / n
    scaled_values = [int(v / scale) for v in values]

    # Inicializa DP
    max_scaled_value = sum(scaled_values)
    dp = [float('inf')] * (max_scaled_value + 1)
    dp[0] = 0

    # Tabela para reconstruir a solução
    item_choice = [[0] * n for _ in range(max_scaled_value + 1)]

    # Preenche a tabela DP
    for i in range(n):
        v = scaled_values[i]
        w = weights[i]
        for total_v in range(max_scaled_value, v - 1, -1):
            if dp[total_v - v] + w <= capacity and dp[total_v - v] + w < dp[total_v]:
                dp[total_v] = dp[total_v - v] + w
                item_choice[total_v] = item_choice[total_v - v][:]
                item_choice[total_v][i] = 1

    # Busca melhor solução viável
    best_scaled_value = max(v for v in range(len(dp)) if dp[v] <= capacity)
    total_value = sum(values[i] for i in range(n) if item_choice[best_scaled_value][i] == 1)

    # Medição de memória
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2

    return total_value, item_choice[best_scaled_value], time.time() - start_time, memory


def branch_and_bound(values, weights, capacity):
    """
    Branch and Bound:
    - Ordena itens por valor/peso.
    - Usa DFS com poda por estimativa de bound.
    """
    n = len(values)
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    best_value = 0.0
    best_solution = [0] * n
    start_time = time.time()
    process = psutil.Process(os.getpid())

    def bound(index, current_weight, current_value):
        """
        Estima bound otimista com relaxação fracionária.
        """
        remaining_capacity = capacity - current_weight
        bound_value = current_value

        for i in range(index, n):
            item = items[i]
            if weights[item] <= remaining_capacity:
                remaining_capacity -= weights[item]
                bound_value += values[item]
            else:
                # Fraciona último item para bound
                bound_value += values[item] * (remaining_capacity / weights[item])
                break
        return bound_value

    def dfs(index, current_weight, current_value, solution):
        nonlocal best_value, best_solution

        # Verifica se excedeu capacidade
        if current_weight > capacity:
            return

        # Atualiza melhor solução
        if current_value > best_value:
            best_value = current_value
            best_solution = solution[:]

        # Fim da lista
        if index >= n:
            return

        # Poda se bound não melhorar solução
        if bound(index, current_weight, current_value) <= best_value:
            return

        item = items[index]

        # Inclui item
        solution[item] = 1
        dfs(index + 1, current_weight + weights[item], current_value + values[item], solution)

        # Exclui item
        solution[item] = 0
        dfs(index + 1, current_weight, current_value, solution)

    # Inicia busca
    dfs(0, 0.0, 0.0, [0] * n)
    total_time = time.time() - start_time
    memory = process.memory_info().rss / 1024 ** 2

    return best_value, best_solution, total_time, memory


def backtracking(values, weights, capacity):
    """
    Backtracking:
    - Explora todas as combinações via DFS.
    - Sem poda por bound.
    """
    n = len(values)
    best_value = 0
    best_solution = [0] * n
    start_time = time.time()
    process = psutil.Process(os.getpid())

    def dfs(index, current_weight, current_value, solution):
        nonlocal best_value, best_solution

        if current_weight > capacity:
            return

        if current_value > best_value:
            best_value = current_value
            best_solution = solution[:]

        if index >= n:
            return

        # Inclui o item
        solution[index] = 1
        dfs(index + 1, current_weight + weights[index], current_value + values[index], solution)

        # Exclui o item
        solution[index] = 0
        dfs(index + 1, current_weight, current_value, solution)

    dfs(0, 0, 0, [0] * n)
    total_time = time.time() - start_time
    memory = process.memory_info().rss / 1024 ** 2

    return best_value, best_solution, total_time, memory
