import os

os.environ["PROJECT_DIR"] = os.path.dirname(os.path.realpath(__file__))
import argparse
import torch

from torch.backends import cudnn
from utils.utils import *

# ours
from solver import Solver

# other MMs
# from other_solver import Solver


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == "train":
        solver.train(training_type="first_train")
    elif config.mode == "test":
        solver.test()
    elif config.mode == "memory_initial":
        solver.get_memory_initial_embedding(training_type="second_train")

    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optimiser
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temp_param", type=float, default=0.05)
    parser.add_argument("--lambd", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    # dataset
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--data_path", type=str, default="./dataset/adult/")
    parser.add_argument("--output_c", type=int, default=1)
    parser.add_argument("--model_save_path", type=str, default="checkpoints")

    # model config
    parser.add_argument(
        "--n_memory", type=int, default=10, help="number of memory items"
    )
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--temperature", type=int, default=0.1)

    # often changes
    parser.add_argument(
        "--mode",
        type=str,
        default="memory_initial",
        choices=["train", "test", "memory_initial"],
    )

    config = parser.parse_args()
    args = vars(config)
    print("------------ Options -------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")
    main(config)
