# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
from joblib import Parallel, delayed
from math import sqrt


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    with Parallel(n_jobs=16) as parallel:
        accumulator = 0.
        n_iter = 0
        while accumulator < 1000:
            results = parallel(delayed(sqrt)(accumulator + i ** 2) for i in range(5))
            print(results)
            accumulator += sum(results)  # synchronization barrier
            n_iter += 1

    print(accumulator)
    print(n_iter)