import os
import time
import sys
import psutil
import csv
import gc
import argparse
import multiprocessing
import threading

# ============================
# Funções de Leitura
# ============================

def read_instance(csv_file: str):
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

def read_text(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        n, capacity = map(float, lines[0].split())
        items = [tuple(map(float, line.split())) for line in lines[1:]]
        values, weights = zip(*items)
        return list(values), list(weights), capacity

def get_optimal(instance_name, opt_directory, directory):
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

def new_2approx(values, weights, capacity):
    n = len(values)
    if n == 0 or capacity <= 0:
        return 0, [0] * n, 0, 0
    items = sorted(range(n), key=lambda i: (values[i] / weights[i]) if weights[i] != 0 else float('inf'), reverse=True)
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

def fptas(values, weights, capacity, epsilon):
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

def branch_and_bound(values, weights, capacity):
    n = len(values)
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    best_value = 0.0
    best_solution = [0] * n
    start_time = time.time()
    process = psutil.Process(os.getpid())

    def bound(index, current_weight, current_value):
        remaining_capacity = capacity - current_weight
        bound_value = current_value
        for i in range(index, n):
            item = items[i]
            if weights[item] <= remaining_capacity:
                remaining_capacity -= weights[item]
                bound_value += values[item]
            else:
                bound_value += values[item] * (remaining_capacity / weights[item])
                break
        return bound_value

    def dfs(index, current_weight, current_value, solution):
        nonlocal best_value, best_solution
        if current_weight > capacity:
            return
        if current_value > best_value:
            best_value = current_value
            best_solution = solution[:]
        if index >= n:
            return
        if bound(index, current_weight, current_value) <= best_value:
            return
        item = items[index]
        solution[item] = 1
        dfs(index + 1, current_weight + weights[item], current_value + values[item], solution)
        solution[item] = 0
        dfs(index + 1, current_weight, current_value, solution)

    dfs(0, 0.0, 0.0, [0] * n)
    total_time = time.time() - start_time
    memory = process.memory_info().rss / 1024 ** 2
    return best_value, best_solution, total_time, memory

def backtracking(values, weights, capacity):
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
        # Escolhe o item
        solution[index] = 1
        dfs(index + 1, current_weight + weights[index], current_value + values[index], solution)
        # Não escolhe o item
        solution[index] = 0
        dfs(index + 1, current_weight, current_value, solution)

    dfs(0, 0, 0, [0] * n)
    total_time = time.time() - start_time
    memory = process.memory_info().rss / 1024 ** 2
    return best_value, best_solution, total_time, memory

# ============================
# Execução de Algoritmos Individuais
# ============================

def run_2approx(instance_name, values, weights, capacity, bb_optimal):
    gc.collect()
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=start_timer, args=(f"[{instance_name} - 2-Approx]", stop_event))
    timer_thread.start()

    try:
        start_time = time.time()
        approx_value, approx_solution, approx_time, approx_memory = new_2approx(values, weights, capacity)
        total_time = time.time() - start_time
    except KeyboardInterrupt:
        stop_event.set()
        timer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print(f"\n[{instance_name}] Execução interrompida pelo usuário (Ctrl+C) durante 2-Approx.\n")
        raise

    stop_event.set()
    timer_thread.join()

    approx_factor = (bb_optimal / approx_value) if (bb_optimal is not None and approx_value > 0) else None

    print(f"[{instance_name}]\t         2-Approx: \tVal = {approx_value:10.2f},   Time = {approx_time:6.2f}s,   Mem = {approx_memory:6.2f}MB,   Approx = {approx_factor:.2f}")

    return {
        'Instance': instance_name,
        'Algorithm': '2-Approx',
        'Value': approx_value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': approx_time,
        'Memory (MB)': approx_memory
    }

def run_fptas(instance_name, values, weights, capacity, epsilon, bb_optimal):
    gc.collect()
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=start_timer, args=(f"[{instance_name} - FPTAS ε={epsilon}]", stop_event))
    timer_thread.start()

    try:
        start_time = time.time()
        fptas_value, fptas_solution, fptas_time, fptas_memory = fptas(values, weights, capacity, epsilon)
        total_time = time.time() - start_time
    except KeyboardInterrupt:
        stop_event.set()
        timer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print(f"\n[{instance_name}] Execução interrompida pelo usuário (Ctrl+C) durante FPTAS (ε={epsilon}).\n")
        raise

    stop_event.set()
    timer_thread.join()

    approx_factor = (bb_optimal / fptas_value) if (bb_optimal is not None and fptas_value > 0) else None

    print(f"[{instance_name}]\t FPTAS (eps={epsilon:.2f}): \tVal = {fptas_value:10.2f},   Time = {fptas_time:6.2f}s,   Mem = {fptas_memory:6.2f}MB,   Approx = {approx_factor:.2f}")

    return {
        'Instance': instance_name,
        'Algorithm': f'FPTAS (eps={epsilon})',
        'Value': fptas_value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': fptas_time,
        'Memory (MB)': fptas_memory
    }

def run_branch_and_bound(instance_name, values, weights, capacity, bb_optimal):
    gc.collect()

    # Inicia o cronômetro em uma thread separada
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=start_timer, args=(f"[{instance_name} - BB]", stop_event))
    timer_thread.start()

    try:
        start_time = time.time()
        bb_value, bb_solution, bb_time, bb_memory = branch_and_bound(values, weights, capacity)
        total_time = time.time() - start_time
    except KeyboardInterrupt:
        stop_event.set()
        timer_thread.join()
        # Limpa a linha pendente
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print(f"\n[{instance_name}] Execução interrompida pelo usuário (Ctrl+C) durante BB.\n")
        raise  # Repassa o KeyboardInterrupt para o processo pai (se quiser manter o comportamento padrão)

    # Para o cronômetro
    stop_event.set()
    timer_thread.join()

    approx_factor = (bb_optimal / bb_value) if (bb_optimal is not None and bb_value > 0) else None

    print(f"[{instance_name}]\t   Branch & Bound: \tVal = {bb_value:10.2f},   Time = {bb_time:6.2f}s,   Mem = {bb_memory:6.2f}MB,   Approx = {approx_factor:.2f}")

    return {
        'Instance': instance_name,
        'Algorithm': f'BB',
        'Value': bb_value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': bb_time,
        'Memory (MB)': bb_memory
    }

def run_backtracking(instance_name, values, weights, capacity, bb_optimal):
    import gc
    gc.collect()

    # Inicia o cronômetro em uma thread separada
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=start_timer, args=(f"[{instance_name} - Backtracking]", stop_event))
    timer_thread.start()

    try:
        start_time = time.time()
        bt_value, bt_solution, bt_time, bt_memory = backtracking(values, weights, capacity)
        total_time = time.time() - start_time
    except KeyboardInterrupt:
        stop_event.set()
        timer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print(f"\n[{instance_name}] Execução interrompida pelo usuário (Ctrl+C) durante Backtracking.\n")
        raise

    # Para o cronômetro
    stop_event.set()
    timer_thread.join()

    # Calcula fator de aproximação se possível
    approx_factor = (bb_optimal / bt_value) if (bb_optimal is not None and bt_value > 0) else None

    print(f"[{instance_name}]\t     Backtracking: \tVal = {bt_value:10.2f},   Time = {bt_time:6.2f}s,   Mem = {bt_memory:6.2f}MB,   Approx = {approx_factor:.2f}")

    return {
        'Instance': instance_name,
        'Algorithm': 'Backtracking',
        'Value': bt_value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': bt_time,
        'Memory (MB)': bt_memory
    }

# ============================
# Processamento de uma Instância
# ============================

def start_timer(message, stop_event):
    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        sys.stdout.write(f"\r{message} Run Time: {elapsed:.0f}s")
        sys.stdout.flush()
        time.sleep(1)
    # Limpa a linha ao final
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

def process_instance(filename, directory, opt_directory, single_instance, result_queue):
    try:
        full_path = os.path.join(directory, filename)

        print("\n" + "=" * 50)
        if filename.endswith('_items.csv'):
            instance_name = filename.replace('_items.csv', '')
            if single_instance and instance_name != single_instance:
                return
            print(f"[{instance_name}] LARGE SCALE")
            values, weights, capacity = read_instance(full_path)
        elif not filename.endswith('.csv'):
            instance_name = filename
            if single_instance and instance_name != single_instance:
                return
            print(f"[{instance_name}] LOW DIMENSIONAL")
            values, weights, capacity = read_text(full_path)
        else:
            return

        bb_optimal = get_optimal(instance_name, opt_directory, directory)
        if bb_optimal is not None:
            print(f"[{instance_name}]\t \t\t\tOpt = {bb_optimal:10.2f}")

        # === Executa 2-Approx ===
        result = run_2approx(instance_name, values, weights, capacity, bb_optimal)
        result_queue.put(result)

        # === Executa FPTAS com diferentes epsilons ===
        for epsilon in [3, 2, 1, 0.5, 0.25]:
            result = run_fptas(instance_name, values, weights, capacity, epsilon, bb_optimal)
            result_queue.put(result)

        # === Executa Branch-and-Bound ===
        result = run_branch_and_bound(instance_name, values, weights, capacity, bb_optimal)
        result_queue.put(result)

        # === Executa Backtracking ===
        result = run_backtracking(instance_name, values, weights, capacity, bb_optimal)
        result_queue.put(result)

    except Exception as e:
        print(f"[{filename}] ERRO: {e}")

# ============================
# Avaliação de Instâncias
# ============================

def evaluate_instances(directory, opt_directory = None, single_instance = None):
    results = []
    TIME_LIMIT = 180  # Tempo limite de 30 minutos

    for filename in os.listdir(directory):
        is_large_scale = filename.endswith('_items.csv')
        is_low_dimensional = not filename.endswith('.csv')
        if not (is_large_scale or is_low_dimensional):
            continue

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=process_instance, args=(filename, directory, opt_directory, single_instance, result_queue))
        p.start()
        p.join(TIME_LIMIT)

        if p.is_alive():
            instance_name = filename.replace('_items.csv', '') if is_large_scale else filename
            print(f"\n[{instance_name}] TIMEOUT: Excedeu o limite de {int(TIME_LIMIT/60)} minutos. Processo abortado.")
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
