import numpy as np
import random
import tensorflow as tf

from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
from src.util.helper import record_result
from src.util.helper import print_result

from src.train import run_netket
from src.offshelf.MaxCut import off_the_shelf
from src.offshelf.manopt_maxcut import manopt
from src.ReptileDemo.sinewave import sinewave

from src.util.Input.InputOutput import read_or_write
from src.util.helper import param_reader

# Quantum learner performs a single QAOA optimization using the framework of chocie.
def quantum_learner(cf, seed, params = np.array([])):
    # set up directories
    #prepare_dirs_and_logger(cf)

    # set up data
    if cf.framework != 'reptile_demo':
        data = read_or_write(cf)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))
    if cf.framework in ["netket"]:
        exp_name, score, time_elapsed, exact_score, params = run_netket(cf, data, seed, params)
    elif cf.framework in ["random_cut", "greedy_cut", "goemans_williamson"]:
        exp_name, score, time_elapsed = off_the_shelf(cf, laplacian=data, method=cf.framework)
        exact_score = 'N/A'
    elif cf.framework in ["manopt"]:
        exp_name, score, time_elapsed, bound = manopt(cf, laplacian=data)
        exact_score = 'N/A'
    elif cf.framework in ['reptile_demo']:
        exp_name, score, time_elapsed, exact_score, params = sinewave(cf, seed, params)
    else:
        raise Exception("unknown framework")
    return exp_name, score, time_elapsed, bound, exact_score, params

# Performs meta-training of quantum_learning using Reptile
def metalearner(cf, seed):
    params = np.array([])
    for num_trials in range(cf.num_trials):
        paramlist = []
        for i in range(cf.metabatch_size):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

            exp_name, score, time_elapsed, bound, exact_score, params2 = quantum_learner(cf,seed,params)
            print_result(cf, exp_name, score, time_elapsed, exact_score)
            paramlist.append(params2)
            print('Metatraining iteration: ' + str(num_trials + 1) + ', batch element: ' + str(i + 1))
            seed += 1
        
        # Metatraining algorithm has to retrieve random initialization from text file on first iteration
        if num_trials == 0:
            params = param_reader(cf)
        
        params2 = np.mean(paramlist, axis = 0)
        params = np.add(params,np.add(params2, params*(-1.0))*float(cf.metalearning_rate))

    param_reader(cf, params)
    return 'finished'


# Performs testing of a previously meta-trained initalization on random samples.
def metatester(cf, seed):
    params = param_reader(cf)
    for trials in range(seed, seed + cf.num_trials):
        np.random.seed(trials)
        random.seed(trials)
        tf.random.set_seed(trials)

        exp_name, score, time_elapsed, bound, exact_score, params2 = quantum_learner(cf,trials,params)
        print_result(cf, exp_name, score, time_elapsed, exact_score)
    return 'finished'