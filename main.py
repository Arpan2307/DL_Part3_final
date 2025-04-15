import json
import argparse
import logging
import matplotlib.pyplot as plt
from trainer import train
import copy


def main():
    # Set up argument parsing and logging
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    temperatures = [0.08, 0.2, 0.7]
    results = run_temperature_experiments(args, temperatures)
    plot_temperature_vs_forgetting(results)


def load_json(settings_path):
    try:
        with open(settings_path) as data_file:
            return json.load(data_file)
    except FileNotFoundError:
        logging.error(f"Config file not found: {settings_path}")
        exit(1)


def setup_parser():
    parser = argparse.ArgumentParser(description='Run contrastive learning temperature experiments.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    return parser


def run_temperature_experiments(args, temperatures):
    results = []
    for temp in temperatures:
        # Create a deep copy of args to avoid modifying the original
        args_temp = copy.deepcopy(args)
        args_temp["contrast_T"] = temp
        logging.info(f"\nRunning experiment with temperature: {temp}")
        avg_forgetting = train(args_temp)
        logging.info(f"Temperature: {temp}, Average Forgetting: {avg_forgetting:.2f}")
        results.append((temp, avg_forgetting))
    return results


def plot_temperature_vs_forgetting(results):
    temperatures, forgetting = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, forgetting, marker='o', linestyle='-', color='b')
    plt.title('Temperature vs. Average Forgetting')
    plt.xlabel('Contrastive Temperature')
    plt.ylabel('Average Forgetting')
    plt.grid(True)
    plt.tight_layout()
    save_path = "temperature_vs_forgetting.png"
    plt.savefig(save_path)
    logging.info(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
