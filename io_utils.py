import os
import csv

def read_instance(csv_file):
    """
    Lê uma instância em formato CSV (items.csv + info.csv) para problemas large-scale.
    """
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


def read_text(file_path):
    """
    Lê instância low-dimensional em formato de texto.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        n, capacity = map(float, lines[0].split())
        items = [tuple(map(float, line.split())) for line in lines[1:]]
        values, weights = zip(*items)
        return list(values), list(weights), capacity


def get_optimal(instance_name, opt_directory, directory):
    """
    Lê o valor ótimo conhecido de uma instância.
    """
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


def parse_instance_name(instance_name):
    """
    Extrai o número de itens e a capacidade máxima do nome da instância.
    """
    parts = instance_name.split('_')
    try:
        if instance_name.endswith('_items'):
            # Large-scale padrão: knapPI_1_500_1000_1_items
            n_items = int(parts[2])
            capacity = int(parts[3])
        else:
            # Low-dimensional padrão: f3_l-d_kp_4_20
            n_items = int(parts[-3])
            capacity = int(parts[-2])
        return n_items, capacity
    except (IndexError, ValueError):
        return None, None


def load_instance_file(filename, directory):
    """
    Escolhe automaticamente a função de leitura conforme o sufixo.
    """
    full_path = os.path.join(directory, filename)
    if filename.endswith('_items.csv'):
        return read_instance(full_path)
    else:
        return read_text(full_path)


def save_results_csv(results, output_csv):
    """
    Salva os resultados em CSV.
    """
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['Instance', 'Items', 'Capacity', 'Algorithm', 'Value',
                       'Optimal', 'Approx Factor', 'Time (s)', 'Memory (MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
