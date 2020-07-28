
import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
from src.util.helper import record_result
from src.util.helper import print_result

from src.train import run_netket
from src.offshelf.MaxCut import off_the_shelf
from src.offshelf.manopt_maxcut import manopt
from RL.train import train

from src.util.Input.InputOutput import read_or_write


def main(cf, seed):
    # set up directories
    prepare_dirs_and_logger(cf)

    # set up data
    data = read_or_write(cf)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))
    if cf.framework in ["netket"]:
        exp_name, score, time_elapsed, exact_score = run_netket(cf, data, seed)
    elif cf.framework in ["random_cut", "greedy_cut", "goemans_williamson"]:
        exp_name, score, time_elapsed = off_the_shelf(cf, laplacian=data, method=cf.framework)
        exact_score = 'N/A'
    elif cf.framework in ["manopt"]:
        exp_name, score, time_elapsed, bound = manopt(cf, laplacian=data)
        exact_score = 'N/A'
    elif cf.framework in ["RL"]:
        exp_name, score, time_elapsed = train(cf, data)
        exact_score = 'N/A'
    else:
        raise Exception("unknown framework")
    return exp_name, score, time_elapsed, bound, exact_score


if __name__ == '__main__':
    cf, unparsed = get_config()
    for num_trials in range(cf.num_trials):
        seed = cf.random_seed + num_trials
        np.random.seed(seed)
        random.seed(seed)

        exp_name, score, time_elapsed, bound, exact_score = main(cf, seed)
        print_result(cf, exp_name, score, time_elapsed, exact_score)

        #Result recorder needs to be rewritten to accomodate new output format
        #record_result(cf, exp_name, score, time_elapsed, bound)
    print('finished')