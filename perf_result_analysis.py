import pandas as pd

perf = pd.read_excel('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/perf_summary_eq_tvwap.xlsx',
                     index_col=0)  # 这边路径记得改

perf = perf[['portfolio_annualized_return', 'benchmark_annualized_return',
             'beta', 'alpha', 'volatility', 'sharp_ratio',
             'information_ratio', 'max_drawdown', 'return_down_ration', 'dd_tdays',
             'H_portfolio_annualized_return', 'H_benchmark_annualized_return',
             'H_beta', 'H_alpha', 'H_volatility', 'H_sharp_ratio',
             'H_information_ratio', 'H_max_drawdown', 'H_return_down_ration', 'H_dd_tdays',
             'turnover']]

# 各大类统计指标平均值
mf = perf[perf.index.str.contains('moneyflow')].mean()
vp = perf[perf.index.str.contains('pricevolume')].mean()
hfvp = perf[perf.index.str.contains('intraday_vp')].mean()
hfmf = perf[~perf.index.str.contains('moneyflow|pricevolume|intraday_vp')].mean()
summa_mean = pd.concat([mf, vp, hfvp, hfmf], keys=['moneyflow', 'pricevolume', 'intraday_vp', 'intraday_moneyflow'], axis=1)

# 各大类统计指标中位数
mf = perf[perf.index.str.contains('moneyflow')].median()
vp = perf[perf.index.str.contains('pricevolume')].median()
hfvp = perf[perf.index.str.contains('intraday_vp')].median()
hfmf = perf[~perf.index.str.contains('moneyflow|pricevolume|intraday_vp')].median()
summa_median = pd.concat([mf, vp, hfvp, hfmf], keys=['moneyflow', 'pricevolume', 'intraday_vp', 'intraday_moneyflow'], axis=1)

# 各大类统计指标75%分位数
mf = perf[perf.index.str.contains('moneyflow')].quantile(0.75)
vp = perf[perf.index.str.contains('pricevolume')].quantile(0.75)
hfvp = perf[perf.index.str.contains('intraday_vp')].quantile(0.75)
hfmf = perf[~perf.index.str.contains('moneyflow|pricevolume|intraday_vp')].quantile(0.75)
summa_75th = pd.concat([mf, vp, hfvp, hfmf], keys=['moneyflow', 'pricevolume', 'intraday_vp', 'intraday_moneyflow'], axis=1)

# 各大类战胜benchmark的比例
a = perf[perf.index.str.contains('moneyflow')]
print('moneyflow_win_ratio: ', len(a[a['portfolio_annualized_return'] > a['benchmark_annualized_return']]) / len(a))
a = perf[perf.index.str.contains('pricevolume')]
print('pricevolume_win_ratio: ', len(a[a['portfolio_annualized_return'] > a['benchmark_annualized_return']]) / len(a))
a = perf[perf.index.str.contains('intraday_vp')]
print('intraday_vp_win_ratio: ', len(a[a['portfolio_annualized_return'] > a['benchmark_annualized_return']]) / len(a))
a = perf[~perf.index.str.contains('moneyflow|pricevolume|intraday_vp')]
print('intraday_moneyflow_win_ratio: ', len(a[a['portfolio_annualized_return'] > a['benchmark_annualized_return']]) / len(a))

summa_mean.to_csv('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/summa_mean.csv')
summa_median.to_csv('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/summa_median.csv')
summa_75th.to_csv('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/summa_75th.csv')
