import json
import argparse
import logging
import matplotlib.pyplot as plt
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    temperatures = [0.08, 0.2, 0.7]
    results = run_temperature_experiments(args, temperatures)
    plot_temperature_vs_forgetting(results)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Run contrastive learning temperature experiments.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    return parser


def run_temperature_experiments(args, temperatures):
    results = []
    for temp in temperatures:
        args["contrast_T"] = temp
        logging.info(f"\nRunning experiment with temperature: {temp}")
        avg_forgetting = train(args)
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
    plt.savefig("temperature_vs_forgetting.png")
    plt.show()


if __name__ == "__main__":
    main()
