import pandas as pd
from alpha_portfolios import portfolio_optimization as po
from alpha_portfolios import backtest_dailyreturns as bd
from utils_func import query_data
from alpha_portfolios import config as cfg
from ft_platform.utils import utils_calculation as uc
import time

begin_date = '2017-12-27'  # 记得修改, 01-05, 02-08, 04-07, 07-05, 12-27
end_date = '2020-08-31'  # 记得修改
data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster/random_forest/window_240'  # 这边路径记得改
pm_pat = 'eq_tvwap'  # 记得修改
fac_data = pd.read_pickle(data_pat + '/fac.pkl')  # 记得修改

# 组合权重设置（一）：使用优化函数
def get_opm_weight_individual(signal=pd.DataFrame(), start_date='2017-01-01', end_date='2020-08-31',
                              out_path='E:/FT_Users/LihaiYang/Files/factor_comb_data/all_cluster_comb/1_pm.csv'):
    pm = po.Portfolio_Management(signal, start_date=start_date, end_date=end_date, num_holdings=cfg.PARAMS_PM['num_holdings'],
                                 style_neutral=False, industry_neutral=True, to_bound=cfg.PARAMS_PM['to_bound'])
    tdays = query_data.get_trade_days('d', from_trade_day=start_date, to_trade_day=end_date)
    newdf = pd.DataFrame()
    w_b, expo = pm.solve(tdays[1])
    newdf = newdf.append(w_b)
    for tday in tdays[2:]:
        print(tday)
        w_b11, expo1 = pm.solve(tday, w_lastday=w_b)
        w_b = w_b11.copy()
        newdf = newdf.append(w_b)
    newdf.index = tdays[1:]
    newdf.to_csv(out_path)
    print(w_b.head())


# 组合权重设置（二）：多头等权重
def get_equal_weight_individual(signal=pd.DataFrame(), start_date='2017-01-01', end_date='2020-08-31',
                                out_path='E:/FT_Users/LihaiYang/Files/factor_comb_data/all_cluster_comb/1_eq.csv'):
    signal = signal[(signal.index >= start_date) & (signal.index <= end_date)]
    weight = (uc.cs_rank(signal) >= 0.9).astype(int)
    weight = weight.div(weight.sum(axis=1), axis=0)
    weight = weight.where(weight > 0)
    weight = weight.dropna(axis=1, how='all')
    weight.to_csv(out_path)


# 组合权重获取
def get_opm_weight(signal):
    for fac in signal.keys():
        get_opm_weight_individual(signal[fac], begin_date, end_date, data_pat + '/' + pm_pat + '/' + str(fac) + '_pm.csv')


def get_equal_weight(signal):
    for fac in signal.keys():
        get_equal_weight_individual(signal[fac], begin_date, end_date, data_pat + '/' + pm_pat + '/' + str(fac) + '_eq.csv')


# 策略回测
def strategy_backtest_individual(opm_weight_path, out_file):
    df_port = pd.read_csv(opm_weight_path, index_col=0)
    df_port.columns = [str(int(v)).zfill(6) for v in df_port.columns]
    df_port.index = pd.to_datetime(df_port.index)
    df_port = (df_port.T / df_port.T.sum()).T
    df = df_port.stack().reset_index()
    df.columns = ['trade_date', 'code', 'weight']
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    init_account = 10000000.0
    df['amt'] = init_account * df['weight']
    df = df[df['amt'] > 0]
    portfolios = df.copy()
    # pro_bt = bd.BackTest(portfolios, fields_trade_prices_buy='stock_open', fields_trade_prices_sell='stock_open')
    pro_bt = bd.BackTest(portfolios, fields_trade_prices_buy='stock_twap_0930_1030', fields_trade_prices_sell='stock_twap_0930_1030')  # 记得修改
    pro_bt.run(filename=out_file)


def strategy_backtest_opm(signal):
    for fac in signal.keys():
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        strategy_backtest_individual(data_pat + '/' + pm_pat + '/' + str(fac) + '_pm.csv', fac + '_pm')


def strategy_backtest_eq(signal):
    for fac in signal.keys():
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        strategy_backtest_individual(data_pat + '/' + pm_pat + '/' + str(fac) + '_eq.csv', fac + '_eq')


# get_opm_weight(fac_data)
# strategy_backtest_opm(fac_data)
get_equal_weight(fac_data)
strategy_backtest_eq(fac_data)

# fac_data = pd.read_pickle('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_wang_20170101-20210228/all_fac_wang1_20170101-20210228.pkl')
# len(fac_data)
# len([len(v) for k, v in fac_data.items() if len(v) == 1009])
# [k for k, v in fac_data.items() if len(v) < 1009]
