import pandas as pd
import os
import json
from utils_func import query_data
from ft_platform.factor_process import fetch
import time

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

with open(data_pat + "/fac_structure.json",'r') as f:
    fac_structure = json.load(f)

mine_summary = query_data.get_alphafactors_info(user='LihaiYang')
mine_fac = [a['factor_name'] for a in mine_summary]

# 找出与自己因子相关性较高的其它人的因子
corr_choose = ['[0.9, 1.0)', '[0.5, 0.9)', '[-0.9, -0.5)', '[-1.0, -0.9)']  # 相关性区间选择
other_fac = {'hfmf': {}, 'mf': {}, 'vp': {}, 'hfvp': {}}
other_fac_all = []
for type, v in fac_structure.items():
    for tag, fac_names in v.items():
        other_fac[type][tag] = []
        for fac_name in fac_names:
            summ = query_data.get_alphafactors_info(factor_name=fac_name)
            if type not in ['hfmf', 'mf']:  # 高频资金流因子没有alpha_information?
                stats_corr = summ[0]['stats_corr']['TOP2000_2017-06-30_2020-06-30']  # 相关系数的时间区间?
                for co, oth in stats_corr.items():
                    if co in corr_choose:
                        other_fac[type][tag].extend([fa for fa in oth[1] if fa not in mine_fac])  # 只纳入除自己因子外的其它人的因子
                        other_fac_all.extend([fa for fa in oth[1] if fa not in mine_fac])

other_fac_all = list(set(other_fac_all))
with open(data_pat + "/other_fac_hcorr.json", "w") as f:
    json.dump(other_fac_all, f)


# 提取与自己因子相关性较高的其它人的因子值
print('fetch')
begin = '2017-01-01'
end = '2021-02-28'
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
factor_value = fetch.fetch_factor(begin, end, fields=other_fac_all, standard='clean1_alla', codes=None, df=False)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 删除因子数据不足的因子
factor_value = {k: v for k, v in factor_value.items() if len(v) == len(query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end))}

# 调整因子的正负

