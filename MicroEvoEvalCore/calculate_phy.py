import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib as Path
from .data_processing import start_test

def start_eval(args):
    path_true = args.path_true
    path_pred = args.path_pred
    data_type = args.data_type
    path = args.save_results
    is_grain = False
    if data_type == 'grain':
        is_grain = True
    start_test(path_true=path_true,path_pred=path_pred, path=path,is_grain=is_grain)
    
    