import argparse
from runner import evaluate_instances

def main():
    parser = argparse.ArgumentParser(
        description="Avaliação de Instâncias de Mochila Binária com Timeout"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        choices=['low', 'large'],
        help="Escolha entre 'low' ou 'large'. Se omitido, executa ambos."
    )
    parser.add_argument(
        '--single',
        type=str,
        help="Nome exato da instância a ser executada"
    )
    args = parser.parse_args()

    DATASETS = {
        "low": {
            "instance_dir": "./low-dimensional",
            "opt_dir": "./low-dimensional-optimum"
        },
        "large": {
            "instance_dir": "./large_scale",
            "opt_dir": None
        }
    }

    datasets_to_run = [args.dataset] if args.dataset else ['low', 'large']
    for dataset_key in datasets_to_run:
        config = DATASETS[dataset_key]
        print(f"\n==> Rodando conjunto: {dataset_key.upper()} (pasta: {config['instance_dir']})")
        evaluate_instances(config['instance_dir'], config['opt_dir'], args.single)


if __name__ == "__main__":
    main()
