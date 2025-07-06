import sys
import time

def start_timer(message, stop_event):
    """
    Mostra um contador de tempo no terminal até o evento de parada ser setado.
    """
    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        sys.stdout.write(f"\r{message:>20}\tRun Time: {elapsed:.0f}s")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    sys.stdout.flush()


def print_table_header():
    """
    Imprime o cabeçalho da tabela de resultados.
    """
    print(f"{'Algorithm':>20} {'Eps':>8} {'Value':>8} {'Time (s)':>10} {'Mem (MB)':>10} {'Approx':>8}")
