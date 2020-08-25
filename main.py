import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.helper import record_result
from src.util.helper import print_result

from src.learners import quantum_learner
from src.learners import metalearner
from src.learners import metatester

#Result recorder needs to be rewritten to accomodate new output format
#record_result(cf, exp_name, score, time_elapsed, bound)

def main(cf, seed):
    if cf.metatrain:
        call = metalearner(cf, seed)
    elif cf.metatest:
        call = metatester(cf, seed)
    elif cf.random_example:
        for trials in range(seed, seed + cf.num_trials):
            np.random.seed(trials)
            random.seed(trials)
            tf.random.set_seed(trials)
            exp_name, score, time_elapsed, bound, exact_score, params2 = quantum_learner(cf, trials)
            print_result(cf, exp_name, score, time_elapsed, exact_score)
        call = 'finished'
    else:
        exp_name, score, time_elapsed, bound, exact_score, params2 = quantum_learner(cf, seed)
        print_result(cf, exp_name, score, time_elapsed, exact_score)
        call = 'finished'
    return call


if __name__ == '__main__':
    cf, unparsed = get_config()
    seed = cf.random_seed
    call = main(cf, seed)
    print(call)