import pandas as pd
import os
import json
from utils_func import query_data

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

with open(data_pat + "/fac_structure.json",'r') as f:
    fac_structure = json.load(f)

