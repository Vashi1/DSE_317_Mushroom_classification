#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:39:19 2022

@author: Tanmay Basu
"""

from regression2 import regression

# from regression3 import regression

import warnings
warnings.filterwarnings("ignore")


rgr=regression('/home/tanmay/ml_class_aug_dec_2023/code/', rgr_opt='rg',
                        no_of_selected_features=4)

rgr.regression()

