import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = args["seed"] if isinstance(args["seed"], list) else [args["seed"]]
    device = copy.deepcopy(args["device"])

    avg_forgetting_all_seeds = []

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = copy.deepcopy(device)
        avg_forgetting = _train(args)
        avg_forgetting_all_seeds.append(avg_forgetting)

    return sum(avg_forgetting_all_seeds) / len(avg_forgetting_all_seeds)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    log_dir = args["log_dir"]
    logs_name = os.path.join(
        log_dir,
        args["model_name"],
        args["dataset"],
        str(init_cls),
        str(args["increment"]),
        args['log_name']
    )
    os.makedirs(logs_name, exist_ok=True)

    logfilename = os.path.join(
        logs_name,
        f"{args['prefix']}_{args['seed']}_{args['convnet_type']}"
    )

    # Reset logging handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )

    model = factory.get_model(args["model_name"], args)
    count_parameters(model)

    acc_matrix = []

    for task in range(data_manager.nb_tasks):
        logging.info(f"Training Task {task}")
        model.incremental_train(data_manager)

        acc_per_task = []
        for eval_task in range(task + 1):
            acc, _ = model.eval_task(eval_task)
            acc_per_task.append(acc["top1"])
            logging.info(f" Eval Task {eval_task} -> Top-1 Acc: {acc['top1']:.2f}")

        acc_matrix.append(acc_per_task)
        model.after_task()

        if args.get("is_task0", False):
            break

    avg_forgetting = compute_average_forgetting(acc_matrix)
    logging.info(f"Average Forgetting: {avg_forgetting:.2f}")
    return avg_forgetting


def compute_average_forgetting(acc_matrix):
    num_tasks = len(acc_matrix)
    forgetting = []

    for task in range(num_tasks - 1):
        acc_list = [acc_matrix[t][task] for t in range(task + 1, num_tasks) if task < len(acc_matrix[t])]
        if not acc_list:
            continue
        max_acc = max([acc_matrix[t][task] for t in range(task + 1) if task < len(acc_matrix[t])])
        final_acc = acc_matrix[-1][task] if task < len(acc_matrix[-1]) else 0
        forgetting.append(max_acc - final_acc)

    return sum(forgetting) / len(forgetting) if forgetting else 0.0


def _set_device(args):
    devs = args["device"]
    gpus = []

    if not isinstance(devs, list):
        devs = [devs]

    for dev in devs:
        if isinstance(dev, torch.device):
            gpus.append(dev)
        elif isinstance(dev, int):
            gpus.append(torch.device("cpu") if dev == -1 else torch.device(f"cuda:{dev}"))
        elif isinstance(dev, str):
            gpus.append(torch.device(dev))
        else:
            raise ValueError(f"Unsupported device type: {type(dev)} — {dev}")

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    logging.info("Arguments:")
    for key, value in args.items():
        logging.info(f"{key}: {value}")
