import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser


def run_temperature_experiments(args, temperatures):
    results = []
    for temp in temperatures:
        args["contrast_T"] = temp
        logging.info(f"Running experiment with temperature: {temp}")
        train(args)
        # Assuming average forgetting is logged or calculated in train
        avg_forgetting = calculate_average_forgetting()  # Placeholder function
        results.append((temp, avg_forgetting))
    return results

# Function to calculate average forgetting based on logged results.
def calculate_average_forgetting():
    # Placeholder logic for calculating average forgetting
    # Replace with actual computation based on task performance logs
    return 0.0  # Example value

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

# Example usage
if __name__ == "__main__":
    with open("./exps/finetune.json", "r") as f:
        args = json.load(f)

    temperatures = [0.08, 0.2, 0.7]
    results = run_temperature_experiments(args, temperatures)
    plot_temperature_vs_forgetting(results)  # Placeholder for plotting function
    
    main()
