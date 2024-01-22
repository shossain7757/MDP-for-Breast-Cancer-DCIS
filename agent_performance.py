import glob
import pandas as pd
import scipy.stats
import numpy as np


def agent_performance(path):
    files = glob.glob(path)
    ls_files = [pd.read_pickle(files) for files in files ]
    y = []
    for exps in ls_files:
        checks = [1 if (tuples[0] % 10 == 0) and (tuples[1] < 8) else 0 for tuples in exps]
        x = round(sum(checks)/len(checks)*100,2)
        y.append(x)

    a = 1.0 * np.array(y)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
    return [m, h]


if __name__ == "__main__":
    dqn = agent_performance("agent_performance/dqn*")
    print(dqn)