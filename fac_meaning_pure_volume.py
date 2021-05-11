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
import json

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/pure_volume/2017-2019'  # 记得修改

# 计算未来1、3、5、10、20日收益率，以开盘1小时tvwap为标准
begin = '2017-01-01'  # 记得修改
end = '2020-02-28'
end1 = '2019-12-31'
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

stock_rela_index = {}
for k in stock_re.keys():
    stock_rela_index[k] = stock_re[k].sub(index_re[k]['000905'], axis=0)

index_re_n = {k: pd.DataFrame(np.tile(v.values, (1, len(list(stock_re[k])))),
                              index=stock_re[k].index, columns=list(stock_re[k])) for k, v in index_re.items()}
"""
# 提取因子数据
list_1 = pd.read_csv(data_pat + '/sector_vp_dailyvolume.csv')
list_2 = pd.read_csv(data_pat + '/sector_vp_intradayvolume.csv')
factor_list = list_1['factor_name'].to_list() + list_2['factor_name'].to_list()
print('fetch')
fac_data = fetch.fetch_factor(begin, end1, fields=factor_list, standard='clean1_alla', codes=None, df=False)
f = open(data_pat + '/fac.pkl', 'wb')
pickle.dump(fac_data, f, -1)
f.close()
"""
# 读取因子数据
fac_data = pd.read_pickle(data_pat + '/fac.pkl')

# top2000股票池
cap_data = fetch_data.fetch(begin, end1, ['stock_tcap'])
cap_rank = cap_data['stock_tcap'].rank(axis=1, ascending=False)
# 每日的top2000股票标记为1，否则为nan
top2000 = (cap_rank <= 2000).where((cap_rank <= 2000) == 1)  # 2015年8月6日只有1999只?

# 根据top2000股票池把因子值在非2000的置为空值
fac_data = {k: (v * top2000) for k, v in fac_data.items()}

"""
# 检测因子  378,377,320  # 特别是高频数据，从15年才有数据，有些因子要用到几个月前的数据?
fac_prob = {k: v for k, v in fac_data.items() if len(v) != len(trade_days)}
for k in fac_data.keys():
    print(k, fac_data[k].T.describe().mean(axis=1))
"""

# 变化符号，扩充因子
fac_expand = {}
for k, v in fac_data.items():
    fac_expand[(k, 1)] = v
    fac_expand[(k, -1)] = -v
del fac_data  # 释放空间

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
            margin = y_bar_margin_test[0][2]  # 这里为何不用y_bar_margin_test?
            r_std = y_bar_margin_test[0][3]  # 添加
            r_sharpe = margin / r_std  # 添加

            if margin < y_bar_margin_max:
                print("挑选完毕,挑选的因子结果如下", list(chosen_xs.keys()))
                break
            else:
                print("    ", "    ", "挑出了最好的因子", x, a, ",绩效为", margin, ",标准差为", r_std, ",夏普比率为", r_sharpe)  # 更改
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
                kkup = pd.DataFrame([x_data_df_test_b.quantile(0.95, axis=1)] * len(x_data_df_test_b.T)).T
                kkup.columns = x_data_df_test_b.columns
                up_df_bool = x_data_df_test_b >= kkup
                margin = round((y_data_concat_df[up_df_bool].mean(axis=1).mean() - y_data_concat_df_index.mean(axis=1).mean()) * 10000, 2)
                s = round(((y_data_concat_df[up_df_bool].mean(axis=1) - y_data_concat_df_index.mean(axis=1)) * 10000).std(), 2)
                if s != 0:
                    y_bar_margin_test.append([x, a, margin, s])
                # print("    ", "    ", "测试第", ii, "个新因子", x, "加入组合,绩效为", margin, s)
            except Exception as e:
                print("测试时", e)

chose_x_func(fac_expand, pd.DataFrame(), data_pat + '/20_d/fac_chosen.json', stock_re['20_d'], index_re_n['20_d'], {}, 0)  # 记得修改
