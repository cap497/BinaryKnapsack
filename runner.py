import os
import sys
import threading
import multiprocessing
import time
import gc

from algorithms import new_2approx, fptas, branch_and_bound, backtracking
from io_utils import get_optimal, parse_instance_name, load_instance_file, save_results_csv
from utils import start_timer, print_table_header

TIME_LIMIT = 1800

# ----------------------------------------
# Impressão
# ----------------------------------------

def format_field(value, width, fmt=None):
    if value is None:
        return ' ' * width
    if value == 'NA':
        return f"{'NA':>{width}}"
    return f"{value:{width}{fmt}}" if fmt else f"{value:{width}}"

def print_result_row(algo_name, eps_value, value, algo_time, memory, approx_factor):
    print(
        f"{algo_name:>20}"
        f" {format_field(eps_value, 8, '.2f')}"
        f" {format_field(value, 8, '.0f')}"
        f" {format_field(algo_time, 10, '.2f')}"
        f" {format_field(memory, 10, '.2f')}"
        f" {format_field(approx_factor, 8, '.2f')}"
    )

def print_instance_banner(instance_type, n_items, capacity_instance, bb_optimal):
    print(f"\n\n{instance_type}\n{'='*100}")
    print(f"N_Items = {n_items}\nW_Max   = {capacity_instance}")
    if bb_optimal is not None:
        print(f"Optimum = {bb_optimal:.0f}")
    print(f"{'='*100}\n")

# ----------------------------------------
# Execução com timer
# ----------------------------------------

def run_algorithm_generic(func, name, instance, values, weights, capacity, bb_optimal, n_items, capacity_instance, extra_args=None, timer_label=None):
    gc.collect()
    stop_event = threading.Event()
    timer = threading.Thread(target=start_timer, args=(timer_label or f"[{name}]", stop_event))
    timer.start()

    try:
        if extra_args:
            result = func(values, weights, capacity, *extra_args)
        else:
            result = func(values, weights, capacity)
        value, solution, algo_time, memory = result
    except KeyboardInterrupt:
        stop_event.set()
        timer.join()
        sys.stdout.write('\r' + ' '*80 + '\r')
        sys.stdout.flush()
        print(f"\nExecução interrompida (Ctrl+C) durante {name}.\n")
        raise
    stop_event.set()
    timer.join()

    approx_factor = (bb_optimal / value) if (bb_optimal is not None and value > 0) else None
    print_result_row(name, extra_args[0] if extra_args else None, value, algo_time, memory, approx_factor)

    return {
        'Instance': instance,
        'Items': n_items,
        'Capacity': capacity_instance,
        'Algorithm': name if not extra_args else f'{name} (eps={extra_args[0]})',
        'Value': value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': algo_time,
        'Memory (MB)': memory
    }

# ----------------------------------------
# Dispatch de algoritmo
# ----------------------------------------

def run_algorithm(algo_name, eps, *args):
    queue = args[-1]
    try:
        mapping = {
            '2-Approx': (new_2approx, "2-Approx", None),
            'FPTAS': (fptas, "FPTAS", [eps]),
            'BB': (branch_and_bound, "Branch & Bound", None),
            'Backtracking': (backtracking, "Backtracking", None)
        }
        if algo_name in mapping:
            func, name, extra = mapping[algo_name]
            res = run_algorithm_generic(
                func, name, *args[:-1], extra_args=extra,
                timer_label=f"[{name} ε={eps}]" if extra else None
            )
            if res:
                queue.put(res)
    except Exception as e:
        print(f"[{args[0]}] ERRO em {algo_name}: {e}")

def run_with_timeout(algo_name, epsilon, *args):
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_algorithm, args=(algo_name, epsilon, *args, result_queue))
    p.start()
    p.join(TIME_LIMIT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    return result_queue.get() if not result_queue.empty() else None

# ----------------------------------------
# Execução de uma instância
# ----------------------------------------

def run_all_algorithms(instance_name, values, weights, capacity, bb_optimal, n_items, capacity_instance, result_queue):
    algorithms = [('2-Approx', None)]
    algorithms += [('FPTAS', eps) for eps in [4, 2, 1, 0.5, 0.25]]
    algorithms += [('BB', None), ('Backtracking', None)]

    for algo_name, eps in algorithms:
        res = run_with_timeout(
            algo_name, eps,
            instance_name, values, weights, capacity,
            bb_optimal, n_items, capacity_instance
        )
        if res is None:
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            print_result_row(algo_name, eps, 'NA', 'NA', 'NA', 'NA')
            result_queue.put({
                'Instance': instance_name,
                'Items': n_items,
                'Capacity': capacity_instance,
                'Algorithm': algo_name if eps is None else f'FPTAS (eps={eps})',
                'Value': 'NA', 'Optimal': bb_optimal,
                'Approx Factor': 'NA', 'Time (s)': 'NA', 'Memory (MB)': 'NA'
            })
        else:
            result_queue.put(res)

def process_instance(filename, directory, opt_directory, single_instance, result_queue):
    try:
        instance_name = filename.replace('_items.csv', '') if filename.endswith('_items.csv') else filename
        n_items, capacity_instance = parse_instance_name(instance_name)
        if single_instance and instance_name != single_instance:
            return

        bb_optimal = get_optimal(instance_name, opt_directory, directory)
        instance_type = "LARGE SCALE" if filename.endswith('_items.csv') else "LOW DIMENSIONAL"
        print_instance_banner(instance_type, n_items, capacity_instance, bb_optimal)

        values, weights, capacity = load_instance_file(filename, directory)
        print_table_header()
        print()

        run_all_algorithms(instance_name, values, weights, capacity,
                           bb_optimal, n_items, capacity_instance, result_queue)

        print()
    except Exception as e:
        print(f"[{filename}] ERRO: {e}")

# ----------------------------------------
# Loop principal
# ----------------------------------------

def evaluate_instances(directory, opt_directory=None, single_instance=None):
    results = []
    all_files = []

    for filename in os.listdir(directory):
        if filename.endswith('_items.csv') or not filename.endswith('.csv'):
            instance_name = filename.replace('_items.csv', '') if filename.endswith('_items.csv') else filename
            n_items, capacity_instance = parse_instance_name(instance_name)
            all_files.append(((n_items or float('inf'), capacity_instance or float('inf'), instance_name), filename))

    for _, filename in sorted(all_files):
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=process_instance,
            args=(filename, directory, opt_directory, single_instance, result_queue)
        )
        p.start()
        p.join()

        while not result_queue.empty():
            results.append(result_queue.get())

    output_csv = f"results_{os.path.basename(directory)}.csv"
    save_results_csv(results, output_csv)
    print(f"\n\nResultados salvos em '{output_csv}'\n\n")
