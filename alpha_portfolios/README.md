# Portfolio

主要为两部分，一部分是组合优化，包括条件约束和不同的目标函数选择；另一部分是，组合收益的计算，用来对照模拟组合和实盘交易的差异


```
python 环境： 使用因子挖掘平台的环境，E:\Share\Alpha\FLi\env\alpha37\Scripts
portfolio_optimization.py  组合条件约束，按照给定的约束条件调整组合中个股的权重
backtest_dailyreturns.py   按照组合中个股权重，生成历史回测收益率
```

## 股票池的约束

股票池跟踪的是指数，只能是每天循环的时候筛选，按照最新的成分股进行调整

## 收益率计算

```
初始化信息
config 文件中调整默认的交易费用等信息

portfolios: pd.DataFrame  历史每个交易日的持仓组合  sort_index， index为交易日 columns = ['code', 'weight', 'amt']
account: int 10000000 账户初始资金
start_date: 回测起始日期, str, default None
        值为空时取portfolios的最小日期
end_date: 回测结束日期, str, default None
        值为空时取portfolios的最大日期
fields_trade_prices_buy: 买入价格 default 'stock_vwap_30m' 支持1、5、30、60、120间隔，30分钟均价
bar_trade_prices_buy: int default 1 从开盘开始第几个bar交易
fields_trade_prices_sell: 卖出价格 default 'stock_vwap_30m'
bar_trade_prices_sell: int default 1 从开盘开始第几个bar交易,注意实盘调仓时尽量先卖后买，所以bar选取要考虑实际
portfolios_create: 组合生成时间 格式'T' 或者 'T-1' 默认T-1  当参数为 T时，应该注意选择交易价格在组合生成之后
tday_lag 交易日延迟 default 1 表示在portfolios_create日期的基础上延后多少个交易日，设置为0时，表示当天交易， 配合portfolios_create使用
trade_mode 交易模式 default: 'target_port' 调整到目标权重 'buy_diff' 卖出不在名单中的个股，买入新入选的个股，持有的个股保持仓位不变
```

## 组合优化

```
初始化信息，在类的说明中

config 文件中的约束条件等配置信息

df_target: pd.DataFrame  个股打分，如果是收益率，则要考虑交易日的移位
start_date: str 起始日期
end_date: str 截止日期
num_holdings: int default 100 目标持股数量，只有当实际持股数超过目标持股数一定范围才起作用
industry_neutral:  行业中性化, str, list or bool, default None 只中性化单个或多个行业时，可输入包含这些行业名称的列表；输入True时，对所有行业进行中性化
style_neutral: 风格中性化 str, list or bool, default False 不做风格约束 输入True时，对所有风格进行中性化
style_expourse: 风格暴露 str, list or bool, default None, 在配置文件中有风格暴露的参数
Epsilon: 中性化约束阈值
st_or_not: 是否包含ST的个股，bool default False 剔除ST
upper_bound: 个股权重上限, float, default None
lower_bound: 个股权重下限, float, default None
to_bound: 换手率约束, float, default None
te_bound: 跟踪误差约束, float, default None 注意，须有因子和个股风险的数据，否则会报错，下同

```