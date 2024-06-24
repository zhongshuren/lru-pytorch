"""
Mostly copied from https://github.com/NicolasZucchet/minimal-LRU/blob/main/run_train.py
"""
import argparse
from dataloading import Datasets


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir_name",
    type=str,
    default="./cache_dir",
    help="name of directory where data is cached",
)
parser.add_argument(
    "--dataset", type=str, choices=Datasets.keys(), default="copy-classification"
)

# Model Parameters
parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the network")
parser.add_argument("--d_model", type=int, default=256, help="Number of features")
parser.add_argument("--d_hidden", type=int, default=128, help="Latent size of recurent unit")
parser.add_argument(
    "--pooling",
    type=str,
    default="none",
    choices=["mean", "last", "none"],
    help="options: (for classification tasks) \\"
         "mean: mean pooling \\"
         "last: take last element \\"
         "none: no pooling",
)
parser.add_argument("--r_min", type=float, default=0.0, help="|lambda|_min for LRU")
parser.add_argument("--r_max", type=float, default=1.0, help="|lambda|_max for LRU")
parser.add_argument("--norm", type=str, default="layer", help="Type of normalization")

# Optimization Parameters
parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
parser.add_argument("--epochs", type=int, default=20, help="Max number of epochs")
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=1000,
    help="Number of epochs to continue training when val loss plateaus",
)
parser.add_argument("--lr_base", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--lr_min", type=float, default=1e-7, help="Minimum learning rate")
parser.add_argument("--lr_factor", type=float, default=0.5, help="ssm lr = lr_factor * lr_base")
parser.add_argument("--cosine_anneal", type=str2bool, default=True, help="Use cosine annealing")
parser.add_argument("--warmup_end", type=int, default=0, help="When to end linear warmup")
parser.add_argument(
    "--lr_patience", type=int, default=1000000, help="Patience before decaying lr"
)
parser.add_argument("--reduce_factor", type=float, default=1.0, help="Factor to decay lr")
parser.add_argument("--p_dropout", type=float, default=0.1, help="Probability of dropout")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay value")