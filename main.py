import json
import argparse
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # Run temperature experiments
    temperatures = [0.08, 0.2, 0.7]
    results = run_temperature_experiments(args, temperatures)
    plot_temperature_vs_forgetting(results)  # Plot the results of the experiments


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json', help='Json file of settings.')
    return parser


def run_temperature_experiments(args, temperatures):
    results = []
    for temp in temperatures:
        args["contrast_T"] = temp
        logging.info(f"Running experiment with temperature: {temp}")
        avg_forgetting = train(args)  # Now this returns avg_forgetting
        results.append((temp, avg_forgetting))
    return results


# Function to plot temperature vs. average forgetting.
def plot_temperature_vs_forgetting(results):
    import matplotlib.pyplot as plt

    temperatures, forgetting = zip(*results)
    plt.plot(temperatures, forgetting, marker='o')
    plt.title('Temperature vs. Average Forgetting')
    plt.xlabel('Temperature')
    plt.ylabel('Average Forgetting')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
