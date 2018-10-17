# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import mimic.sense_inventory.pyumls.api as api

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    # x = api.getByCUI('C0927232', apikey='52295d6a-08a6-4ecc-86fd-8239857fb64f')
    x = api.getByCUI('C3714787', apikey='52295d6a-08a6-4ecc-86fd-8239857fb64f')
    print(x)