from ft_platform import fetch_data
from ft_platform.utils import utils_calculation as uc
import pandas as pd
import pickle
from utils_func import query_data
from ft_platform.factor_process import fetch
import json
from copy import deepcopy
import numpy as np
import time
import os

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

# 计算未来1、3、5、10、20日收益率，以开盘1小时tvwap为标准
begin = '2017-01-01'  # 记得修改
end = '2020-10-31'  # 记得修改
end1 = '2020-08-31'  # 记得修改
data = fetch_data.fetch(begin, end, ['stock_adjtwap_0930_1030'])
index_data = fetch_data.fetch(begin, end, ['index_close'], '000905')
stock_re = {}
stock_re['1_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -2) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['3_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -4) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['5_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -6) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['10_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -11) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['20_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -21) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end1)
stock_re = {k: v.loc[trade_days] for k, v in stock_re.items()}
index_re = {}
index_re['1_d'] = uc.ts_delay(index_data['index_close'], -1) / index_data['index_close'] - 1
index_re['3_d'] = uc.ts_delay(index_data['index_close'], -3) / index_data['index_close'] - 1
index_re['5_d'] = uc.ts_delay(index_data['index_close'], -5) / index_data['index_close'] - 1
index_re['10_d'] = uc.ts_delay(index_data['index_close'], -10) / index_data['index_close'] - 1
index_re['20_d'] = uc.ts_delay(index_data['index_close'], -20) / index_data['index_close'] - 1
index_re = {k: v.loc[trade_days] for k, v in index_re.items()}

index_re_n = {k: pd.DataFrame(np.tile(v.values, (1, len(list(stock_re[k])))),
                              index=stock_re[k].index, columns=list(stock_re[k])) for k, v in index_re.items()}

# 读取因子数据
fac_old = pd.read_pickle(data_pat + '/fac_last.pkl')
fac_fundamental = {}
pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/fundamental/'
for way in ['all_eq', '50%_eq', 'sharpe_weight']:
    for j in os.listdir(pat + way):
        if j[-3:] == 'pkl':
            temp = pd.read_pickle(pat + way + '/' + j)
            fac_fundamental = dict(fac_fundamental, **temp)
fac_fundamental = {k: v for k, v in fac_fundamental.items() if k in ['50%_eq_fundamental_growth',
                                                                     '50%_eq_fundamental_earning',
                                                                     'sharpe_weight_fundamental_valuation']}
fac_data = dict(fac_old, **fac_fundamental)  # 不做标准化应该也没事，后面用的是rank等权聚合

# 变化符号，扩充因子
fac_pos = {(k, 1): v for k, v in fac_data.items()}
fac_neg = {(k, -1): -v for k, v in fac_data.items()}
fac_expand = {**fac_pos, **fac_neg}


def chose_x_func(wait_delete_xs: dict,
                 x_data_df: pd.DataFrame,
                 chosen_xs_file: str,
                 y_data_concat_df: pd.DataFrame,
                 y_data_concat_df_index: pd.DataFrame,
                 chosen_xs: dict,
                 y_bar_margin: int = 0) -> None:
    """
    :param wait_delete_xs:         等待选择的因子字典, key为(x的名字, x的系数1或-1), value为x的值DataFrame(index为日期,columns为股票代码)
                                   e.g { ("factor_100001_vp", 1): pd.DataFrame(xxxxxx) }
    :param x_data_df:              已经选择的因子组合值的DataFrame(index为日期,columns为股票代码)(默认为空) e.g pd.DataFrame()
    :param chosen_xs_file:         保存因子组合的文件路径 e.g r"./combo/10_days/chosen.json"
    :param y_data_concat_df:       个股return的值(index为日期,columns为股票代码)
    :param y_data_concat_df_index: 指数return的值(index为日期,columns为股票代码同上,但是元素值用每天的指数return来替换每天的个股return)
    :param chosen_xs:              已经选择的因子字典(默认为空) e.g {}, 如果已经选好了n个初始因子则{ "factor_100001_vp": 1}
    :param y_bar_margin:           平均的超额指数的每个股票的n天周期return的数值,默认初始为0
    :return: None
    """
    """
    1.首先如果有已选择进组合的因子,那就先把这些因子组合起来,算出一个当前margin(没有就是0)
    2.然后从wait_delete_xs里一个个因子加进现有的组合里测试,挑出加进去后效果最好的,只要比之前组合绩效好就加进组合,写入组合.json文件, 并把它从wait_delete_xs里删除.
    3.然后继续从wait_delete_xs里一个个因子测,并重复步骤2,直到效果最好的因子加进去后新组合绩效仍然比老的组合绩效差,就退出整个流程.
    """

    print("##########开始测试新的因子##########")
    print("一共 %d 个因子" % (len(wait_delete_xs.keys())))
    y_bar_margin_test = list()
    x_data_df_test = x_data_df.copy()
    y_bar_margin_max = 0
    count = 0
    while len(wait_delete_xs.keys()) > 0:

        count += 1
        if len(y_bar_margin_test) == 0:
            # 初始第一个因子
            x_data_df_test = x_data_df.copy()
            y_bar_margin_max = y_bar_margin
        else:
            # 开始组合第二个因子
            # 先对上次的结果逆序,拿到组合后最好的那个因子
            y_bar_margin_test.sort(key=lambda piece: piece[2], reverse=True)
            sorted_xs = deepcopy(y_bar_margin_test)
            x = sorted_xs[0][0]
            a = sorted_xs[0][1]
            margin = y_bar_margin_test[0][2]

            if margin < y_bar_margin_max:
                print("挑选完毕,挑选的因子结果如下", list(chosen_xs.keys()))
                break
            else:
                print("    ", "    ", "挑出了最好的因子", x, a, ",绩效为", margin)
                # 选好了因子,然后把因子值加入组合
                x_data_concat_df = wait_delete_xs[(x, a)]

                if len(x_data_df_test) == 0:
                    x_data_df_test = x_data_concat_df.rank(axis=1)
                else:
                    x_data_df_test = x_data_df_test.copy() + x_data_concat_df.rank(axis=1)
                # 然后完结
                y_bar_margin_max = margin
                if x not in chosen_xs.keys():
                    chosen_xs[x] = a
                else:
                    chosen_xs[x] += a
                with open(chosen_xs_file, 'w') as f:
                    json.dump(chosen_xs, fp=f, indent=4)
                f.close()
                wait_delete_xs.pop((x, a))

        ############################################################################################################
        print("    ", "测试第", count, "轮")
        y_bar_margin_test = list()
        """
        下面这个每层的循环遍历组合绩效,以后可以改多进程计算来加速
        """
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for ii, (x, a) in enumerate(wait_delete_xs.keys()):
            try:
                x_data_df_test_b = x_data_df_test.copy()
                x_data_concat_df = wait_delete_xs[(x, a)]

                if len(x_data_df_test_b) == 0:
                    x_data_df_test_b = x_data_concat_df.rank(axis=1)
                else:
                    x_data_df_test_b = x_data_df_test_b.copy() + x_data_concat_df.rank(axis=1)
                kkup = pd.DataFrame([x_data_df_test_b.quantile(0.9, axis=1)] * len(x_data_df_test_b.T)).T  # 记得修改
                kkup.columns = x_data_df_test_b.columns
                up_df_bool = x_data_df_test_b >= kkup
                margin = round((y_data_concat_df[up_df_bool].mean(axis=1).mean() - y_data_concat_df_index.mean(axis=1).mean()) * 10000, 2)
                s = round(((y_data_concat_df[up_df_bool].mean(axis=1) - y_data_concat_df_index.mean(axis=1)) * 10000).std(), 2)
                if s != 0:
                    y_bar_margin_test.append([x, a, margin, s])
                print("    ", "    ", "测试第", ii, "个新因子", x, "加入组合,绩效为", margin, s)
            except Exception as e:
                print("测试时", e)

chose_x_func(fac_expand, pd.DataFrame(), data_pat + '/fac_chosen.json', stock_re['1_d'], index_re_n['1_d'], {}, 0)  # 记得修改
