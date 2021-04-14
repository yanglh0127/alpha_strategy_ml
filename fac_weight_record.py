import pandas as pd
import os
import copy
import json

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data'

fac_cluster = pd.read_excel(data_pat + '/fac_meaning/all_cluster/all.xlsx', sheet_name='各类聚合因子的表现', index_col=0)
fac_hierarchy = {}
fac_hierarchy['hfmf'] = {'高频资金流分布': ['日内资金流分布', '收盘资金流行为', '开盘资金流行为', '中间资金流行为'],
                         '大单行为': ['平均单笔成交金额', '大单资金流向'], '反转因子改进': ['反转因子改进']}
fac_hierarchy['mf'] = {'日间资金流波动': ['开盘资金流的日间波动', '收盘资金流的日间波动', '资金流的日间波动',
                                   '主力净流入的绝对值', '主力净流入的时间趋势绝对值'],
                       '主力单数行为': ['主力单数行为'], '主力流入流出占比': ['主力流入占比', '主力流出占比'],
                       '反转因子改进': ['反转因子改进'], '价格和资金流向的相关性': ['价格和资金流向的相关性'],
                       '开盘净主动买入行为': ['开盘净主动买入行为', '开盘和收盘净主动买入之差'],
                       '主力净流入占比的偏度': ['主力净流入占比的偏度'], '收盘主力净流入行为': ['收盘主力净流入行为']}
fac_hierarchy['hfvp'] = {'日内成交额分布的稳定性': ['日内成交额的波动', '日内成交额的偏度', '日内成交额的峰度'],
                         '收盘行为异常': ['收盘行为异常'], '日内成交额的自相关': ['日内成交额的自相关'],
                         '反转因子相关': ['反转因子相关'], '流动性因子改进': ['流动性因子改进'],
                         '日内收益率的分布': ['波动率的波动率', '尾部风险', '日内收益率的偏度', '日内收益率的波动率'],
                         '高频贝塔': ['高频贝塔'], '日内不同时段成交量差异': ['上午下午开盘成交量差异', '日内中间时段成交量占比'],
                         '高频量价相关性': ['高频量价相关性'], '隔夜(或上午)和下午收益率差异': ['隔夜(或上午)和下午收益率差异'],
                         '高频收益率为正和负时的波动率差异': ['高频收益率为正和负时的波动率差异']}
fac_hierarchy['vp'] = {'日间成交量(额)的波动率': ['日间成交量(额)的波动率'], '反转因子相关': ['反转因子相关'],
                       '量价相关性': ['量价相关性'], '收益率和波动率的相关性': ['收益率和波动率的相关性'],
                       '情绪因子': ['情绪因子'], '流动性因子相关': ['流动性因子相关']}
fac_structure = copy.deepcopy(fac_hierarchy)
fac_weight = {}


def cal_weight(data, way):
    weight = {}
    if 'all' in way:
        weight = {su[15:-3]: 1 for su in data.index}
    if '50%' in way:
        sub_data = data[data['top_50%'] == 1]
        weight = {su[15:-3]: 1 for su in sub_data.index}
    if 'sharpe' in way:
        sub_data = data[data['positive'] == 1]
        weight = {su[15:-3]: sub_data.loc[su, 'sharp_ratio'] for su in sub_data.index}
    if 'best' in way:
        weight = {data['sharp_ratio'].idxmax()[15:-3]: 1}
    return weight


for i in fac_cluster.index:
    if i.split('_')[-1] == 'hfmf':
        fac_meaning = pd.read_excel(data_pat + '/all_fac_20170101-20210228/fac_meaning.xlsx', sheet_name='高频资金流向', index_col=0)
        fac_meaning['top_50%'] = (fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5)
        fac_meaning['positive'] = (fac_meaning['sharp_ratio'] > 0)
        tag = i.split('_')[-2]
        sub = fac_meaning[fac_meaning['tag1'].isin(fac_hierarchy['hfmf'][tag])]
        temp_name = [su[15:-3] for su in sub.index]
        fac_structure['hfmf'][tag] = temp_name
        fac_weight[i] = cal_weight(sub, i.split('_')[0])
    if i.split('_')[-1] == 'mf':
        fac_meaning = pd.read_excel(data_pat + '/all_fac_20170101-20210228/fac_meaning.xlsx', sheet_name='日频资金流向', index_col=0)
        fac_meaning['top_50%'] = (fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5)
        fac_meaning['positive'] = (fac_meaning['sharp_ratio'] > 0)
        tag = i.split('_')[-2]
        sub = fac_meaning[fac_meaning['tag1'].isin(fac_hierarchy['mf'][tag])]
        temp_name = [su[15:-3] for su in sub.index]
        fac_structure['mf'][tag] = temp_name
        fac_weight[i] = cal_weight(sub, i.split('_')[0])
    if i.split('_')[-1] == 'hfvp':
        fac_meaning = pd.read_excel(data_pat + '/all_fac_20170101-20210228/fac_meaning.xlsx', sheet_name='高频量价', index_col=0)
        fac_meaning['top_50%'] = (fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5)
        fac_meaning['positive'] = (fac_meaning['sharp_ratio'] > 0)
        tag = i.split('_')[-2]
        sub = fac_meaning[fac_meaning['tag1'].isin(fac_hierarchy['hfvp'][tag])]
        temp_name = [su[15:-3] for su in sub.index]
        fac_structure['hfvp'][tag] = temp_name
        fac_weight[i] = cal_weight(sub, i.split('_')[0])
    if i.split('_')[-1] == 'vp':
        fac_meaning = pd.read_excel(data_pat + '/all_fac_20170101-20210228/fac_meaning.xlsx', sheet_name='日频量价', index_col=0)
        fac_meaning['top_50%'] = (fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5)
        fac_meaning['positive'] = (fac_meaning['sharp_ratio'] > 0)
        tag = i.split('_')[-2]
        sub = fac_meaning[fac_meaning['tag1'].isin(fac_hierarchy['vp'][tag])]
        temp_name = [su[15:-3] for su in sub.index]
        fac_structure['vp'][tag] = temp_name
        fac_weight[i] = cal_weight(sub, i.split('_')[0])

with open(data_pat + "/fac_meaning/all_cluster/fac_hierarchy.json", "w") as f:
    json.dump(fac_hierarchy, f)
with open(data_pat + "/fac_meaning/all_cluster/fac_structure.json", "w") as f:
    json.dump(fac_structure, f)
with open(data_pat + "/fac_meaning/all_cluster/fac_weight.json", "w") as f:
    json.dump(fac_weight, f)
