import os
import time
import psutil
import csv
import gc
import argparse
import multiprocessing
from typing import List, Tuple, Optional

# ============================
# Funções de Leitura de Instâncias
# ============================

def read_knapsack_instance(csv_file: str) -> Tuple[List[int], List[int], int]:
    values, weights = [], []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Arquivo CSV {csv_file} não contém cabeçalho.")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for row in reader:
            values.append(int(row['price'].strip()))
            weights.append(int(row['weight'].strip()))

    info_file = csv_file.replace('items.csv', 'info.csv')
    with open(info_file, 'r') as f:
        reader = csv.reader(f)
        capacity = None
        for row in reader:
            if row and row[0].strip() == 'c':
                capacity = int(row[1])
                break
    if capacity is None:
        raise ValueError(f"Capacidade (c) não encontrada em {info_file}")
    return values, weights, capacity

def read_knapsack_text(file_path: str) -> Tuple[List[int], List[int], int]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        n, capacity = map(int, lines[0].split())
        items = [tuple(map(lambda x: int(float(x)), line.split())) for line in lines[1:]]
        values, weights = zip(*items)
        return list(values), list(weights), capacity

def get_optimal_solution(instance_name: str, opt_directory: Optional[str], directory: str) -> Optional[float]:
    if opt_directory:
        opt_file = os.path.join(opt_directory, instance_name)
        if os.path.exists(opt_file):
            with open(opt_file, 'r') as f:
                return float(f.read().strip())
    else:
        # Caso large_scale, procurar no info.csv
        info_file = os.path.join(directory, f"{instance_name}_info.csv")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip() == 'z':
                        return float(row[1])
    return None

# ============================
# Algoritmos
# ============================

def branch_and_bound_knapsack(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int], float, float]:
    n = len(values)
    best_value = 0
    best_solution = [0] * n

    def knapsack_helper(value: int, weight: int, index: int, solution: List[int]):
        nonlocal best_value, best_solution
        if weight <= capacity and value > best_value:
            best_value = value
            best_solution = solution[:]
        if index >= n or weight >= capacity:
            return
        knapsack_helper(value, weight, index + 1, solution)
        if weight + weights[index] <= capacity:
            solution[index] = 1
            knapsack_helper(value + values[index], weight + weights[index], index + 1, solution)
            solution[index] = 0

    initial_solution = [0] * n
    start_time = time.time()
    knapsack_helper(0, 0, 0, initial_solution)
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2
    return best_value, best_solution, time.time() - start_time, memory

def new_2approx_knapsack(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int], float, float]:
    n = len(values)
    if n == 0 or capacity <= 0:
        return 0, [0] * n, 0, 0
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    total_value = 0
    solution = [0] * n
    remaining_capacity = capacity
    start_time = time.time()
    for i in items:
        if weights[i] <= remaining_capacity:
            solution[i] = 1
            total_value += values[i]
            remaining_capacity -= weights[i]
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2
    return total_value, solution, time.time() - start_time, memory

def fptas_knapsack(values: List[int], weights: List[int], capacity: int, epsilon: float) -> Tuple[int, List[int], float, float]:
    n = len(values)
    start_time = time.time()

    vmax = max(values)
    scale = epsilon * vmax / n
    scaled_values = [int(v / scale) for v in values]

    max_scaled_value = sum(scaled_values)
    dp = [float('inf')] * (max_scaled_value + 1)
    dp[0] = 0
    item_choice = [[0] * n for _ in range(max_scaled_value + 1)]

    for i in range(n):
        v = scaled_values[i]
        w = weights[i]
        for total_v in range(max_scaled_value, v - 1, -1):
            if dp[total_v - v] + w <= capacity and dp[total_v - v] + w < dp[total_v]:
                dp[total_v] = dp[total_v - v] + w
                item_choice[total_v] = item_choice[total_v - v][:]
                item_choice[total_v][i] = 1

    best_scaled_value = max(v for v in range(len(dp)) if dp[v] <= capacity)
    total_value = sum(values[i] for i in range(n) if item_choice[best_scaled_value][i] == 1)

    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2
    return total_value, item_choice[best_scaled_value], time.time() - start_time, memory

# ============================
# Processamento de uma Instância
# ============================

def format_approx(factor):
    return f"{factor:.4f}" if factor is not None else "N/A"

def process_single_instance(filename: str, directory: str, opt_directory: Optional[str], single_instance: Optional[str], result_queue):
    try:
        full_path = os.path.join(directory, filename)

        print("\n" + "=" * 50)
        if filename.endswith('_items.csv'):
            instance_name = filename.replace('_items.csv', '')
            if single_instance and instance_name != single_instance:
                return
            print(f"[{instance_name}] Lendo instância LARGE SCALE...")
            values, weights, capacity = read_knapsack_instance(full_path)
        elif not filename.endswith('.csv'):
            instance_name = filename
            if single_instance and instance_name != single_instance:
                return
            print(f"[{instance_name}] Lendo instância LOW DIMENSIONAL...")
            values, weights, capacity = read_knapsack_text(full_path)
        else:
            return

        bb_optimal = get_optimal_solution(instance_name, opt_directory, directory)
        if bb_optimal is not None:
            print(f"[{instance_name}] Ótimo conhecido: {bb_optimal}")

        # === Novo 2-Approx ===
        print(f"[{instance_name}] Iniciando Novo 2-Aproximativo...")
        gc.collect()
        approx_value, approx_solution, approx_time, approx_memory = new_2approx_knapsack(values, weights, capacity)
        approx_approx_factor = (bb_optimal / approx_value) if bb_optimal and approx_value > 0 else None
        result_queue.put({
            'Instance': instance_name,
            'Algorithm': '2-Approx',
            'Value': approx_value,
            'Optimal': bb_optimal,
            'Approx Factor': approx_approx_factor,
            'Time (s)': approx_time,
            'Memory (MB)': approx_memory
        })
        print(f"[{instance_name}] Resultado - 2-Approx: Valor={approx_value}, Tempo={approx_time:.2f}s, Memória={approx_memory:.2f}MB, Aproximação={format_approx(approx_approx_factor)}")

        # === FPTAS com diferentes eps ===
        for epsilon in [2.0, 1.0, 0.5, 0.1]:
            print(f"[{instance_name}] Iniciando FPTAS com eps={epsilon}...")
            gc.collect()
            fptas_value, fptas_solution, fptas_time, fptas_memory = fptas_knapsack(values, weights, capacity, epsilon)
            fptas_approx_factor = (bb_optimal / fptas_value) if bb_optimal and fptas_value > 0 else None
            result_queue.put({
                'Instance': instance_name,
                'Algorithm': f'FPTAS (eps={epsilon})',
                'Value': fptas_value,
                'Optimal': bb_optimal,
                'Approx Factor': fptas_approx_factor,
                'Time (s)': fptas_time,
                'Memory (MB)': fptas_memory
            })
            print(f"[{instance_name}] Resultado - FPTAS (eps={epsilon}): Valor={fptas_value}, Tempo={fptas_time:.2f}s, Memória={fptas_memory:.2f}MB, Aproximação={format_approx(fptas_approx_factor)}")

        # === Branch-and-Bound ===
        print(f"[{instance_name}] Iniciando Branch-and-Bound...")
        gc.collect()
        bb_value, bb_solution, bb_time, bb_memory = branch_and_bound_knapsack(values, weights, capacity)
        bb_approx_factor = (bb_optimal / bb_value) if bb_optimal and bb_value > 0 else None
        result_queue.put({
            'Instance': instance_name,
            'Algorithm': 'Branch-and-Bound',
            'Value': bb_value,
            'Optimal': bb_optimal,
            'Approx Factor': bb_approx_factor,
            'Time (s)': bb_time,
            'Memory (MB)': bb_memory
        })
        print(f"[{instance_name}] Resultado - Branch-and-Bound: Valor={bb_value}, Tempo={bb_time:.2f}s, Memória={bb_memory:.2f}MB, Aproximação={format_approx(bb_approx_factor)}")

    except Exception as e:
        print(f"[{filename}] ERRO: {e}")

# ============================
# Avaliação de Instâncias
# ============================

def evaluate_instances(directory: str, opt_directory: Optional[str] = None, single_instance: Optional[str] = None):
    results = []
    TIME_LIMIT = 180  # 3 minutos por instância

    for filename in os.listdir(directory):
        is_large_scale = filename.endswith('_items.csv')
        is_low_dimensional = not filename.endswith('.csv')
        if not (is_large_scale or is_low_dimensional):
            continue

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=process_single_instance, args=(filename, directory, opt_directory, single_instance, result_queue))
        p.start()
        p.join(TIME_LIMIT)

        if p.is_alive():
            print(f"[{filename}] TIMEOUT: Excedeu o limite de {TIME_LIMIT/60} minutos. Processo abortado.")
            p.terminate()
            p.join()
        else:
            while not result_queue.empty():
                results.append(result_queue.get())

    output_csv = f"results_{os.path.basename(directory)}.csv"
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['Instance', 'Algorithm', 'Value', 'Optimal', 'Approx Factor', 'Time (s)', 'Memory (MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n\nResultados salvos em '{output_csv}'\n\n")

# ============================
# Execução
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de Instâncias de Mochila Binária com Timeout")
    parser.add_argument('dataset', nargs='?', choices=['low', 'large'], help="Escolha entre 'low' ou 'large'. Se omitido, executa ambos.")
    parser.add_argument('--single', type=str, help="Nome exato da instância a ser executada")
    args = parser.parse_args()

    DATASETS = {
        "low": {"instance_dir": "./low-dimensional", "opt_dir": "./low-dimensional-optimum"},
        "large": {"instance_dir": "./large_scale", "opt_dir": None}
    }

    datasets_to_run = [args.dataset] if args.dataset else ['low', 'large']
    for dataset_key in datasets_to_run:
        config = DATASETS[dataset_key]
        print(f"\n==> Rodando conjunto: {dataset_key.upper()} (pasta: {config['instance_dir']})")
        evaluate_instances(config['instance_dir'], config['opt_dir'], args.single)
