import pandas as pd
import os

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228_oos'  # 这边路径记得改
weight_way = os.listdir(data_pat)
weight_way.remove('all_fac_20170101-20210228.pkl')  # 记得修改

results_perf = {}
results_hperf = {}
results_to = {}
for ww in weight_way:
    new_dir = os.path.join(data_pat, ww)
    comb_way = [i for i in os.listdir(new_dir) if os.path.isdir(os.path.join(new_dir, i))]
    for cw in comb_way:
        nn_dir = os.path.join(new_dir, cw)
        for j in os.listdir(nn_dir):
            if j[-3:] == 'pkl':
                result = pd.read_pickle(os.path.join(nn_dir, j))
                results_perf[ww + '_' + cw] = result['perf']
                results_hperf[ww + '_' + cw] = result['hedged_perf']
                results_to[ww + '_' + cw] = result['turnover_series'].mean()

perf = pd.concat(results_perf, axis=1)
hperf = pd.concat(results_hperf, axis=1)
hperf.index = 'H_' + hperf.index
to = pd.DataFrame.from_dict(results_to, orient='index')
to.columns = ['turnover']
perf_summary = pd.concat([perf, hperf])
perf_summary = pd.concat([perf_summary.T, to], axis=1)
perf_summary.to_csv(data_pat + '/perf_summary.csv')
