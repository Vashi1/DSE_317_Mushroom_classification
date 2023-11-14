#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:39:19 2022

@author: Tanmay Basu
"""

# from classification2 import classification
from classification3 import classification

import warnings
warnings.filterwarnings("ignore")


clf=classification('/home/tanmay/ml_class_aug_dec_2023/code/', clf_opt='dt',
                        no_of_selected_features=4)

clf.classification()

