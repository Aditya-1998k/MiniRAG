import os
import multiprocessing
import torch


def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
