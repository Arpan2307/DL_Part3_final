import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import json


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
    logs_name = "{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args["increment"], args['log_name'])
    logs_name = os.path.join(log_dir, logs_name)

    os.makedirs(logs_name, exist_ok=True)

    logfilename = os.path.join(log_dir, "{}/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args['log_name'],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    ))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    args["dataset"] = "cifar100"
    args["log_name"] = "experiment_without_pic"

    _set_random()
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

    cnn_curve = {"top1": [], "top5": []}
    task_acc = []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        model.incremental_train(data_manager)
        cnn_accy, _ = model.eval_task()

        top1_acc = cnn_accy["top1"]
        task_acc.append(top1_acc)

        logging.info("Task {} - CNN top1: {:.2f}".format(task, top1_acc))
        cnn_curve["top1"].append(top1_acc)
        cnn_curve["top5"].append(cnn_accy["top5"])

        model.after_task()

        if args["is_task0"]:
            break

    avg_forgetting = compute_average_forgetting(task_acc)
    logging.info(f"Average Forgetting: {avg_forgetting:.2f}")
    return avg_forgetting


def compute_average_forgetting(acc_list):
    forgetting = []
    for i in range(1, len(acc_list)):
        max_prev = max(acc_list[:i])
        forgetting.append(max_prev - acc_list[i])
    return sum(forgetting) / len(forgetting) if forgetting else 0.0


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for dev in device_type:
        if isinstance(dev, torch.device):
            gpu_device = dev
        elif isinstance(dev, str) and dev.startswith("cuda"):
            gpu_device = torch.device(dev)
        elif dev == -1 or dev == "cpu":
            gpu_device = torch.device("cpu")
        elif isinstance(dev, int):
            gpu_device = torch.device(f"cuda:{dev}")
        else:
            raise ValueError(f"Invalid device specifier: {dev}")
        gpus.append(gpu_device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
