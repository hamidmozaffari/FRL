# import parser as _parser

import argparse
import sys
# import yaml

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="FRL")
    
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="Logs",
        help="Location to logs/checkpoints",)
    
    parser.add_argument("--set", type=str, default="CIFAR10" , help="Which dataset to use")
    
    parser.add_argument(
        "--nClients", type=int, default=1000, help="number of clients participating in FL (default: 1000)")
    parser.add_argument(
        "--at_fractions", type=float, default=0.0, help="fraction of malicious clients (default: 0%)")
    
    parser.add_argument(
        "--non_iid_degree",
        type=float,
        default=1.0,
        help="non-iid degree data distribution given to Dirichlet Distribution (default: 1.0)",
    )
   
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    
    parser.add_argument(
        "--data_loc", type=str, default="/scratch/hamid/CIFAR10/", help="Location to store data",
    )
    
    parser.add_argument(
        "--conv_type", type=str, default="MaskConv", help="Type of conv layer (defualt: MaskConv)"
    )
    
    parser.add_argument(
        "--FL_type", type=str, default="FRL", help="Type of FL (defualt: FRL)"
    )
    
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="number of local epochs to train in each FL client (default: 5)",
    )
    parser.add_argument(
        "--FL_global_epochs",
        type=int,
        default=1000,
        help="number of FL global epochs to train the global model (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.4,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--lrdc",
        type=float,
        default=0.999,
        help="learning rate decay (default: 0.999)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )
    
    parser.add_argument("--model", type=str, default="Conv8", help="Type of model (default: Conv8().")
    
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="how sparse is each layer, when using MaskConv"
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    
    parser.add_argument(
        "--round_nclients", type=int, default=25, help="Number of selected clients in each round"
    )
    parser.add_argument(
        "--rand_mal_clients", type=int, default=25, help="Number of selected malicious clients in each round to generate the malicious update"
    )
    parser.add_argument("--name", type=str, default="FRL_no_mal", help="Experiment id.")
    
    parser.add_argument(
        "--config", type=str, default=None, help="Config file to use"
    )
#     parser.add_argument(
#         "--optimizer", type=str, default="sgd", help="Which optimizer to use"
#     )




#     parser.add_argument(
#         "--log-interval",
#         type=int,
#         default=10,
#         metavar="N",
#         help="how many batches to wait before logging training status",
#     )
#     parser.add_argument("--workers", type=int, default=4, help="how many cpu workers")
#     parser.add_argument(
#         "--output-size",
#         type=int,
#         default=10,
#         help="how many total neurons in last layer",
#     )
#     parser.add_argument(
#         "--real-neurons", type=int, default=10, help="how many real neurons"
#     )
#     parser.add_argument("--name", type=str, default="default", help="Experiment id.")
#     parser.add_argument(
#         "--data", type=str, help="Location to store data",
#     )

#     parser.add_argument("--resume", type=str, default=None, help='optionally resume')
#     parser.add_argument(
#         "--sparsity", type=float, default=0.5, help="how sparse is each layer, when using MultitaskMaskConv"
#     )
#     parser.add_argument("--gamma", type=float, default=0.0)
#     parser.add_argument(
#         "--width-mult", type=float, default=1.0, help="how wide is each layer"
#     )
#     parser.add_argument(
#         "--hop-weight", type=float, default=1e-3, help="how wide is each layer"
#     )
#     parser.add_argument(
#         "--conv_type", type=str, default="StandardConv", help="Type of conv layer"
#     )
#     parser.add_argument(
#         "--bn_type", type=str, default="StandardBN", help="Type of batch norm layer."
#     )
#     parser.add_argument(
#         "--conv-init",
#         type=str,
#         default="default",
#         help="How to initialize the conv weights.",
#     )
#     parser.add_argument("--model", type=str, help="Type of model.")
#     parser.add_argument(
#         "--multigpu",
#         default=None,
#         type=lambda x: [int(a) for a in x.split(",")],
#         help="Which GPUs to use for multigpu training",
#     )
#     parser.add_argument(
#         "--eval-ckpts",
#         default=None,
#         type=lambda x: [int(a) for a in x.split(",")],
#         help="After learning n tasks for n in eval_ckpts we perform evaluation on all tasks learned so far",
#     )
#     parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
#     parser.add_argument(
#         "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
#     )

#     parser.add_argument(
#         "--num-tasks",
#         default=None,
#         type=int,
#         help="Number of tasks, None if no adaptation is necessary",
#     )
#     parser.add_argument(
#         "--adaptor",
#         default="gt",
#         help="Which adaptor to use, see adaptors.py",
#     )
#     parser.add_argument("--set", type=str, help="Which dataset to use")
#     parser.add_argument("--er-sparsity", action="store_true", default=False)
#     parser.add_argument(
#         "--trainer",
#         default=None,
#         type=str,
#         help="Which trainer to use, default in trainers/default.py",
#     )
#     parser.add_argument(
#         "--log-base",
#         default=2,
#         type=int,
#         help="keep the bottom 1/log_base elements during binary optimization",
#     )
#     parser.add_argument(
#         "--save", action="store_true", default=False, help="save checkpoints"
#     )
#     parser.add_argument(
#         "--train-weight-tasks",
#         type=int,
#         default=0,
#         metavar="N",
#         help="number of tasks to train the weights, e.g. 1 for batchensembles. -1 for all tasks",
#     )
#     parser.add_argument(
#         "--train-weight-lr",
#         default=0.1,
#         type=float,
#         help="While training the weights, which LR to use.",
#     )

#     parser.add_argument(
#         "--individual-heads",
#         action="store_true",
#         help="Seperate head for each batch_ensembles task!",
#     )
#     parser.add_argument("--no-scheduler", action="store_true", help="constant LR")

#     parser.add_argument(
#         "--iter-lim", default=-1, type=int,
#     )

#     parser.add_argument(
#         "--ortho-group", action="store_true", default=False,
#     )

#     # TODO: task-eval move out to diff main
#     parser.add_argument("--lr-policy", default=None, help="Scheduler to use")
#     parser.add_argument(
#         "--task-eval",
#         default=None,
#         type=int,
#         help="Only evaluate on this task (for memory efficiency and grounded task info",
#     )
#     parser.add_argument(
#         "-f",
#         "--dummy",
#         default=None,
#         help="Dummy to use for ipython notebook compatibility",
#     )

#     parser.add_argument(
#         "--warmup-length", default=0, type=int,
#     )
#     parser.add_argument(
#         "--reinit-most-recent-k",
#         default=None,
#         type=int,
#         help="Whether or not to include a memory buffer for reinit training. Currently only works with binary reinit_adaptor",
#     )
#     parser.add_argument(
#         "--reinit-adapt",
#         type=str,
#         default="binary",
#         help="Adaptor for reinitialization experiments",
#     )

#     parser.add_argument(
#         "--data-to-repeat", default=1, type=int,
#     )

#     parser.add_argument(
#         "--unshared_labels", action="store_true", default=False,
#     )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    """Parses the config file and returns the values of the arguments."""
    load_args={}
    with open(args.config, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value
            load_args[key] = value
    args.__dict__.update(load_args)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()