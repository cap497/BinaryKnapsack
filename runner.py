import sys
import os
import gc
import threading
import multiprocessing
import time

from algorithms import new_2approx, fptas, branch_and_bound, backtracking
from io_utils import (
    read_instance,
    read_text,
    get_optimal,
    parse_instance_name,
    load_instance_file,
    save_results_csv
)
from utils import start_timer, print_table_header

TIME_LIMIT = 1800  # segundos

def run_algorithm_generic(algorithm_func, algo_name, instance_name, values, weights, capacity,
                          bb_optimal, n_items, capacity_instance,
                          extra_args=None, timer_label=None):
    """
    Roda qualquer algoritmo com timer e coleta de memória.
    """
    gc.collect()

    stop_event = threading.Event()
    label = timer_label or f"[{algo_name}]"
    timer_thread = threading.Thread(target=start_timer, args=(label, stop_event))
    timer_thread.start()

    try:
        start_time = time.time()
        if extra_args:
            value, solution, algo_time, memory = algorithm_func(values, weights, capacity, *extra_args)
        else:
            value, solution, algo_time, memory = algorithm_func(values, weights, capacity)
    except KeyboardInterrupt:
        stop_event.set()
        timer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print(f"\nExecução interrompida pelo usuário (Ctrl+C) durante {algo_name}.\n")
        raise

    stop_event.set()
    timer_thread.join()

    approx_factor = (bb_optimal / value) if (bb_optimal is not None and value > 0) else None
    eps_value = extra_args[0] if extra_args else None
    eps_str = f"{eps_value:8.2f}" if eps_value is not None else " " * 8
    print(f"{algo_name:>20} {eps_str} {value:8.0f} {algo_time:10.2f} {memory:10.2f} {approx_factor:8.2f}")

    return {
        'Instance': instance_name,
        'Items': n_items,
        'Capacity': capacity_instance,
        'Algorithm': algo_name if eps_value is None else f'{algo_name} (eps={eps_value})',
        'Value': value,
        'Optimal': bb_optimal,
        'Approx Factor': approx_factor,
        'Time (s)': algo_time,
        'Memory (MB)': memory
    }


def run_algorithm(algo_name, eps, instance_name, values, weights, capacity,
                   bb_optimal, n_items, capacity_instance, queue):
    """
    Wrapper para rodar no multiprocessing.
    """
    try:
        if algo_name == '2-Approx':
            res = run_algorithm_generic(
                new_2approx, "2-Approx", instance_name, values, weights, capacity,
                bb_optimal, n_items, capacity_instance
            )
        elif algo_name == 'FPTAS':
            res = run_algorithm_generic(
                fptas, "FPTAS", instance_name, values, weights, capacity,
                bb_optimal, n_items, capacity_instance,
                extra_args=[eps],
                timer_label=f"[FPTAS ε={eps}]"
            )
        elif algo_name == 'BB':
            res = run_algorithm_generic(
                branch_and_bound, "Branch & Bound", instance_name, values, weights, capacity,
                bb_optimal, n_items, capacity_instance
            )
        elif algo_name == 'Backtracking':
            res = run_algorithm_generic(
                backtracking, "Backtracking", instance_name, values, weights, capacity,
                bb_optimal, n_items, capacity_instance
            )
        else:
            res = None

        if res:
            queue.put(res)
    except Exception as e:
        print(f"[{instance_name}] ERRO em {algo_name}: {e}")


def run_with_timeout(algo_name, epsilon, *args):
    """
    Gerencia subprocesso com timeout.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_algorithm, args=(algo_name, epsilon, *args, result_queue))
    p.start()
    p.join(TIME_LIMIT)

    if p.is_alive():
        p.terminate()
        p.join()
        return None
    elif result_queue.empty():
        return None
    else:
        return result_queue.get()


def process_instance(filename, directory, opt_directory, single_instance, result_queue):
    """
    Processa uma única instância (low ou large scale).
    """
    try:
        instance_name = filename.replace('_items.csv', '') if filename.endswith('_items.csv') else filename
        n_items, capacity_instance = parse_instance_name(instance_name)

        if single_instance and instance_name != single_instance:
            return

        bb_optimal = get_optimal(instance_name, opt_directory, directory)

        num_hyphens = 100
        if filename.endswith('_items.csv'):
            print("\n\nLARGE SCALE")
        else:
            print("\n\nLOW DIMENSIONAL")
        print("=" * num_hyphens)
        print(f"N_Items = {n_items}")
        print(f"W_Max   = {capacity_instance}")
        if bb_optimal is not None:
            print(f"Optimum = {bb_optimal:.0f}")
        print("=" * num_hyphens)
        print()

        values, weights, capacity = load_instance_file(filename, directory)
        print_table_header()
        print()

        # Algoritmos a rodar
        algorithms_to_run = [('2-Approx', None)]
        for eps in [4, 2, 1, 0.5, 0.25]:
            algorithms_to_run.append(('FPTAS', eps))
        algorithms_to_run.extend([('BB', None), ('Backtracking', None)])

        for algo_name, eps in algorithms_to_run:
            res = run_with_timeout(
                algo_name, eps,
                instance_name, values, weights, capacity,
                bb_optimal, n_items, capacity_instance
            )

            if res is None:
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                sys.stdout.flush()
                eps_str = f"{eps:8.2f}" if eps is not None else " " * 8
                print(f"{algo_name:>20} {eps_str} {'NA':>8} {'NA':>10} {'NA':>10} {'NA':>8}")
                result_queue.put({
                    'Instance': instance_name,
                    'Items': n_items,
                    'Capacity': capacity_instance,
                    'Algorithm': algo_name if eps is None else f'FPTAS (eps={eps})',
                    'Value': 'NA',
                    'Optimal': bb_optimal,
                    'Approx Factor': 'NA',
                    'Time (s)': 'NA',
                    'Memory (MB)': 'NA'
                })
            else:
                result_queue.put(res)

        print()

    except Exception as e:
        print(f"[{filename}] ERRO: {e}")


def evaluate_instances(directory, opt_directory=None, single_instance=None):
    """
    Loop principal para avaliar todas as instâncias de um diretório.
    """
    results = []
    all_files = []

    for filename in os.listdir(directory):
        is_large_scale = filename.endswith('_items.csv')
        is_low_dimensional = not filename.endswith('.csv')
        if not (is_large_scale or is_low_dimensional):
            continue

        if is_large_scale:
            instance_name = filename.replace('_items.csv', '')
        else:
            instance_name = filename

        n_items, capacity_instance = parse_instance_name(instance_name)
        sort_key = (n_items if n_items is not None else float('inf'),
                    capacity_instance if capacity_instance is not None else float('inf'),
                    instance_name)
        all_files.append((sort_key, filename))

    all_files.sort()

    for _, filename in all_files:
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
