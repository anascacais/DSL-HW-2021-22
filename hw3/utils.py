# built-in
import random

# third-party
import numpy as np
import matplotlib.pyplot as plt
import torch


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def read_data(filepath, partitions=None):
    """Read the Ar-En dataset."""
    f = open(filepath)
    input_lang = []
    target_lang = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\t\n")
            fields = line.split("\t")
            input_lang += [fields[0]]
            target_lang += [fields[1]]
    return input_lang, target_lang


def plot(epochs, plottable, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.plot(epochs, plottable)
    plt.savefig("{}.png".format(name), bbox_inches="tight")
