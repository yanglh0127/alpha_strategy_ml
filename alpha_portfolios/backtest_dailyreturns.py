#python3

"""
主要实现组合的历史收益率回测
本次主要功能为参照王峰使用的matlab程序，基于python实现(基于杨峰的backtest)
@author: yyshi
"""

import pandas as pd
import numpy as np
import time
import pickle
from collections import OrderedDict
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from alpha_portfolios import config as cfg
from ft_platform.factor_process.fetch import fetch_factor
from ft_platform import fetch_data
from utils_func.utils import *
from utils_func import query_data

def portfolio_performance(portfolio_return, benchmark_index, rf=0.0, freq='d'):
    """
    收益率统计指标， 日频
    :param portfolio_return: 组合收益率 pd.Series
    :param benchmark_index: 基准指数价格 pd.Series
    :param rf: 无风险收益率
    :param freq: 数据频率 日'd' 周'w' 月'm'
    :return:
    """
    return_free = rf  # 无风险利率估算为3%
    d_const_1y = {'d': 252.0, 'w': 50.0, 'm': 12.0}
    const_1y = d_const_1y[freq]
    # 计算组合价值
    portfolio_return.fillna(0, inplace=True)
    portfolio_value = (portfolio_return + 1).cumprod()
    # 计算基准价值和基准收益
    benchmark_value = benchmark_index / benchmark_index[0]
    benchmark_return = benchmark_value.pct_change()

    # 收益年化化
    portfolio_annualized_return = (
        portfolio_value[-1] ** (const_1y / len(portfolio_value)) - 1
    )
    benchmark_annualized_return = (
        benchmark_value[-1] ** (const_1y / len(benchmark_value)) - 1
    )

    # 计算组合的beta和alpha
    beta = portfolio_return.cov(benchmark_return) / benchmark_return.var()
    alpha = (portfolio_annualized_return - return_free) - beta * (
        benchmark_annualized_return - return_free
    )

    # 计算组合波动率
    volatility = portfolio_return.std() * (const_1y ** 0.5)
    # 计算组合夏普值
    sharp_ratio = (portfolio_annualized_return- return_free) / volatility

    # 计算组合追踪误差
    track_err_std = (portfolio_return - beta * benchmark_return).std() * (252 ** 0.5)
    information_ratio = alpha / track_err_std

    # 计算组合最大回撤
    max_drawdown = 1 - min(
        portfolio_value / np.maximum.accumulate(portfolio_value.fillna(-np.inf))
    )

    # 最大回撤
    drawdown = portfolio_value / np.maximum.accumulate(portfolio_value.fillna(-np.inf))
    max_drawdown = 1 - drawdown.min()
    max_drawdown_end = drawdown.idxmin()
    max_drawdown_start = portfolio_value[:max_drawdown_end].idxmax(
    ) if max_drawdown_end > portfolio_value.index[0] else 0
    max_drawdown_date = (max_drawdown_start.strftime('%Y/%m/%d'), max_drawdown_end.strftime('%Y/%m/%d'))

    # 最长亏损时段
    max_cum_val = np.maximum.accumulate(portfolio_value.dropna())
    max_cum_val_uniq = np.unique(max_cum_val, return_counts=True)
    most_occurrence_idx = np.argmax(max_cum_val_uniq[1])
    most_occurrence_val = max_cum_val_uniq[0][most_occurrence_idx]
    most_occurrence_val_pos = np.where(max_cum_val == most_occurrence_val)
    max_drawdown_period = most_occurrence_val_pos[0][-1] - most_occurrence_val_pos[0][0]
    max_drawdown_period_start = portfolio_value.index[most_occurrence_val_pos[0][0]]
    max_drawdown_period_end = portfolio_value.index[most_occurrence_val_pos[0][-1]]
    max_drawdown_period_date = (
    max_drawdown_period_start.strftime('%Y/%m/%d'), max_drawdown_period_end.strftime('%Y/%m/%d'))

    test_start = portfolio_return.index[0]
    test_end = portfolio_return.index[-1]
    test_date = (test_start.strftime('%Y/%m/%d'), test_end.strftime('%Y/%m/%d'))
    trade_days = len(portfolio_return)
    test_period = trade_days

    # 组合性能
    perf = {
        "portfolio_total_return": portfolio_value[-1] - 1,  # 组合收益率
        "portfolio_annualized_return": portfolio_annualized_return,  # 年化收益率
        "benchmark_total_return": benchmark_value[-1] - 1,  # 基准价值
        "benchmark_annualized_return": benchmark_annualized_return,  # 基准年化收益率
        "beta": beta,  # 组合beta
        "alpha": alpha,  # 组合alpha
        "volatility": volatility,  # 组合波动率
        "sharp_ratio": sharp_ratio,  # 组合夏普值
        "information_ratio": information_ratio,  # 组合信息比
        "max_drawdown": max_drawdown,  # 组合最大回撤
        "return_down_ration": portfolio_annualized_return / max_drawdown,  # 收益回撤比
        "dd_interval": max_drawdown_date,  # 最大回撤区间
        "dd_tdays": max_drawdown_period,  # 最大回撤时间长度
        "revovery_period": max_drawdown_period_date,  # 净值回复区间
        "total_tdays": '%s(%s)' % (test_period, freq)  # 回测区间长度
    }
    return pd.Series(perf)

def plot_performance_single(port_returns, benchmark_returns, title, return_free=0.00, freq='d'):
    """
    绘制收益率曲线和收益率各项评价指标。

    Parameters
    ----------
    port_returns : pd.Series 收益率序列
    benchmark_returns: pd.Series 基准收益率
    title : str 图片标题,保存路径
    return_free : float 无风险收益率
    preq: str 收益率的周期
    """

    values = (port_returns + 1).cumprod()

    red = "#aa4643"
    green = '#156b09'
    blue = "#4572a7"
    black = "#000000"

    fig = plt.figure(figsize=(18, 10))
    gs = mpl.gridspec.GridSpec(14, 1)

    font_size = 12
    value_font_size = 14
    label_height, value_height = 0.75, 0.55
    label_height2, value_height2 = 0.30, 0.10

    perf = portfolio_performance(port_returns, benchmark_returns, rf=return_free, freq=freq)
    text_data = [
        (0.05, label_height, value_height, u"AccReturn", "{0:.2%}".format(perf["portfolio_total_return"]), black, red),
        (0.05, label_height2, value_height2, u"I_AccReturn", "{0:.2%}".format(perf["benchmark_total_return"]), black, red),

        (0.17, label_height, value_height, u"AnnulReturn", "{0:.3f}".format(perf["portfolio_annualized_return"]), black, red),
        (0.17, label_height2, value_height2, u"I_AnnulReturn", "{0:.2%}".format(perf["benchmark_annualized_return"]), black, red),

        (0.29, label_height, value_height, u"Sharpe", "{0:.3f}".format(perf["sharp_ratio"]), black, black),
        (0.29, label_height2, value_height2, u"Volatility", "{0:.2%}".format(perf["volatility"]), black, black),

        (0.41, label_height, value_height, u"DrawDown", "{0:.2%}".format(perf["max_drawdown"]), black, black),
        (0.41, label_height2, value_height2, u"DDTdays", "{0}".format(perf["dd_tdays"]), black, black),

        (0.53, label_height, value_height, u"Alpha", "{0:.2%}".format(perf["alpha"]), black, black),
        (0.53, label_height2, value_height2, u"Beta", "{0:.2%}".format(perf["beta"]), black, black),

        (0.65, label_height, value_height, u"IR", "{0:.3f}".format(perf["information_ratio"]), black, black),
        (0.65, label_height2, value_height2, u"SampleDate", "{}".format(perf["total_tdays"]), black, black),

        (0.77, label_height, value_height, u"DDInterval", perf["DDInterval"][0] + ' ~ ' +
         perf["DDInterval"][1], black, black),
        (0.77, label_height2, value_height2, u"RecoveryPeriod", perf["RecoveryPeriod"][0] + ' ~ ' +
         perf["RecoveryPeriod"][1], black, black)
    ]

    ax1 = fig.add_subplot(gs[:3])
    ax1.axis("off")
    for x, y1, y2, label, value, label_color, value_color in text_data:
        ax1.text(x, y1, label, color=label_color, fontsize=font_size)
        ax1.text(x, y2, value, color=value_color, fontsize=value_font_size)
    ax1.set_title(title.split('\\')[-1], fontsize=20)

    ax2 = fig.add_subplot(gs[4:10])
    ax2.plot(values - 1, c=red, lw=2, label=u'NAV')
    ax2.plot(values[list(perf['DDInterval'])] - 1, 'v', color=green, markersize=8, alpha=.7, label=u"DDInterval")
    ax2.plot(values[list(perf['RecoveryPeriod'])] - 1, 'D', color=blue, markersize=8, alpha=.7, label=u"RecoveryPeriod")

    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2%}'))
    ax2.legend(loc='best')
    ax2.grid()

    ax3 = fig.add_subplot(gs[11:])
    returns_positive = pd.Series(np.zeros(len(port_returns)), index=port_returns.index)
    returns_negative = pd.Series(np.zeros(len(port_returns)), index=port_returns.index)
    returns_positive.where(port_returns < 0, port_returns, inplace=True)
    returns_negative.where(port_returns > 0, port_returns, inplace=True)
    ax3.fill_between(returns_positive.index, returns_positive.values, color=red, label=u'Profit')
    ax3.fill_between(returns_negative.index, returns_negative.values, color=green, label=u'Loss')
    ax3.legend(loc='best')
    ax3.spines['top'].set_color('none')
    ax3.spines['right'].set_color('none')

    plt.savefig(title + '.png', bbox_inches='tight')
    # plt.show()

def change_freq(df_ret, mode='ret', to_freq='w'):
    """
    将日频收益率转换成指定频率的收益率
    :param pd.Series 日收益率序列， index为交易日
    :param mode str default ret: 收益率模式， 可选 ('turnover' 换手率， 'nav': 净值)
    :param to_freq : str 'w' weekly  'm' monthly
    :return:
    """
    start_date = df_ret.index[0].strftime('%Y-%m-%d')
    end_date = df_ret.index[-1].strftime('%Y-%m-%d')
    date_f = query_data.get_trade_days(to_freq, from_trade_day=start_date, to_trade_day=end_date)
    date_f.sort()
    if mode=='ret':
        df_nav = (1+df_ret).cumprod()
        newdf_ret = df_nav.loc[date_f].pct_change()
        newdf_ret.iloc[0] = df_nav.loc[date_f[0]] / df_nav.iloc[0]-1
        return newdf_ret
    elif mode=='turnover':
        df_nav = df_ret.cumsum()
        newdf_nav = df_nav.loc[date_f]
        newdf_ret = newdf_nav - newdf_nav.shift(1)
        newdf_ret.iloc[0] = newdf_nav.iloc[0]
        return newdf_ret
    elif mode=='nav':
        return df_ret.loc[date_f]

def plot_portfolio_performance(
    portfolio_return,
    portfolio_turnover,
    hedged_return,
    benchmark_index,
    perf,
    hedged_perf,
    title,
    log_y=None,
    hedged_only=False,
    fig_handler=False,
):
    """
    绘制组合表现
    :param portfolio_return: pd.Series 组合多头收益率
    :param portfolio_turnover: pd.Series 换手率
    :param hedged_return:  pd.Series 组合对冲后收益率
    :param benchmark_index: pd.Series 基准收益率
    :param perf: pd.Series 组合多头表现
    :param hedged_perf: pd.Series 组合对冲表现
    :param title: str
    :param log_y: Bool 是否使用对数收益
    :param hedged_only: Bool True只绘制对冲图像
    :param fig_handler:
    :return:
    """

    # 计算组合收益率，组合价值，基准收益率，基准价值
    portfolio_value = (portfolio_return + 1).cumprod()
    hedged_value = (hedged_return + 1).cumprod()
    benchmark_value = benchmark_index / benchmark_index[0]
    benchmark_return = benchmark_value.pct_change()
    benchmark_return[0] = 0

    # 绘图中文显示
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [u"Microsoft Yahei", u"SimHei", u"sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 颜色编码
    red = "#aa4643"
    green = "#156b09"
    blue = "#4572a7"
    black = "#000000"

    # 图像绘制
    fig = plt.figure(figsize=(18, 14))
    gs = mpl.gridspec.GridSpec(18, 1)
    # 图像尺寸编码
    font_size = 12
    value_font_size = 14
    label_height, value_height = 0.75, 0.55
    label_height2, value_height2 = 0.30, 0.10

    # 记录数据
    text_data = [
        (
            0.00,
            label_height,
            value_height,
            u"AccReturn",
            "{0:.2%}".format(perf["portfolio_total_return"]),
            black,
            red,
        ),
        (
            0.00,
            label_height2,
            value_height2,
            u"AnnulReturn",
            "{0:.2%}".format(perf["portfolio_annualized_return"]),
            black,
            red,
        ),
        (
            0.085,
            label_height,
            value_height,
            "Alpha",
            "{0:.2%}".format(perf["alpha"]),
            black,
            black,
        ),
        (
            0.085,
            label_height2,
            value_height2,
            "Beta",
            "{0:.3f}".format(perf["beta"]),
            black,
            black,
        ),
        (
            0.16,
            label_height,
            value_height,
            "Sharpe",
            "{0:.3f}".format(perf["sharp_ratio"]),
            black,
            black,
        ),
        (
            0.16,
            label_height2,
            value_height2,
            "IR",
            "{0:.3f}".format(perf["information_ratio"]),
            black,
            black,
        ),
        (
            0.24,
            label_height,
            value_height,
            "DrawDown",
            "{0:.2%}".format(perf["max_drawdown"]),
            black,
            black,
        ),
        (
            0.24,
            label_height2,
            value_height2,
            "Volatility",
            "{0:.2%}".format(perf["volatility"]),
            black,
            black,
        ),
        (
            0.37,
            label_height,
            value_height,
            u"AccReturn_H",
            "{0:.2%}".format(hedged_perf["portfolio_total_return"]),
            black,
            blue,
        ),
        (
            0.37,
            label_height2,
            value_height2,
            u"AnnulReturn_H",
            "{0:.2%}".format(hedged_perf["portfolio_annualized_return"]),
            black,
            blue,
        ),
        (
            0.46,
            label_height,
            value_height,
            "Alpha_H",
            "{0:.2%}".format(hedged_perf["alpha"]),
            black,
            black,
        ),
        (
            0.46,
            label_height2,
            value_height2,
            "Beta_H",
            "{0:.3f}".format(hedged_perf["beta"]),
            black,
            black,
        ),
        (
            0.53,
            label_height,
            value_height,
            "Sharpe_H",
            "{0:.3f}".format(hedged_perf["sharp_ratio"]),
            black,
            black,
        ),
        (
            0.53,
            label_height2,
            value_height2,
            "IR_H",
            "{0:.3f}".format(hedged_perf["information_ratio"]),
            black,
            black,
        ),
        (
            0.61,
            label_height,
            value_height,
            "DrawDown_H",
            "{0:.2%}".format(hedged_perf["max_drawdown"]),
            black,
            black,
        ),
        (
            0.61,
            label_height2,
            value_height2,
            "Volatility_H",
            "{0:.2%}".format(hedged_perf["volatility"]),
            black,
            black,
        ),
        (
            0.75,
            label_height,
            value_height,
            "Ret2DD(L/H)",
            "{0:.3f} / {1:.3f}".format(
                perf["return_down_ration"], hedged_perf["return_down_ration"]
            ),
            black,
            black,
        ),
        (
            0.75,
            label_height2,
            value_height2,
            "turnover",
            "{0:.3f}".format(portfolio_turnover.mean()),
            black,
            black,
        ),
        (
            0.90,
            label_height,
            value_height,
            u"AccReturn_I",
            "{0:.2%}".format(perf["benchmark_total_return"]),
            black,
            black,
        ),
        (
            0.90,
            label_height2,
            value_height2,
            u"AnnulReturn_I",
            "{0:.2%}".format(perf["benchmark_annualized_return"]),
            black,
            black,
        ),
    ]

    #
    ax1 = fig.add_subplot(gs[:3])
    ax1.axis("off")
    for x, y1, y2, label, value, label_color, value_color in text_data:
        ax1.text(x, y1, label, color=label_color, fontsize=font_size)
        ax1.text(x, y2, value, color=value_color, fontsize=value_font_size)
    ax1.set_title(title, fontsize=20)

    #
    ax2 = fig.add_subplot(gs[4:10])
    ax2_2 = ax2.twinx()
    ax2.plot(portfolio_value, c=red, lw=2, label=u"PortLong")
    ax2.plot(benchmark_value, c="gray", lw=2, label=u"Index")
    ax2_2.plot(hedged_value, c=blue, lw=2, label=u"Hedge")
    if log_y in ["left", "both"]:
        ax2.set_yscale("log")
    if log_y in ["right", "both"]:
        ax2_2.set_yscale("log")

    #
    ax2.grid(alpha=0.4, ls="dashed", axis="x")
    if hedged_only:
        lines = ax2_2.lines
        labels = ["Hedge"]
        for line in ax2.lines:
            line.set_visible(False)
        ax2.yaxis.set_visible(False)
    else:
        lines = ax2.lines + ax2_2.lines
        labels = ["PortLong", "Index", "Hedge(Right)"]
    plt.grid(alpha=0.4, ls="dashed", axis="y")
    plt.legend(lines, labels)

    #
    ax3 = fig.add_subplot(gs[11:14])
    return_diff = portfolio_return - benchmark_return
    return_diff_positive = pd.Series(
        np.zeros(len(return_diff)), index=return_diff.index
    )
    return_diff_negative = pd.Series(
        np.zeros(len(return_diff)), index=return_diff.index
    )
    return_diff_positive.where(return_diff < 0, return_diff, inplace=True)
    return_diff_negative.where(return_diff > 0, return_diff, inplace=True)
    ax3.fill_between(
        return_diff_positive.index, return_diff_positive.values, color=red, label="Profit"
    )
    ax3.fill_between(
        return_diff_negative.index,
        return_diff_negative.values,
        color=green,
        label="Loss",
    )
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2%}"))
    ax3.legend(loc="upper left")
    ax3.spines["top"].set_color("none")
    ax3.spines["right"].set_color("none")

    #
    ax4 = fig.add_subplot(gs[15:])
    ax4.fill_between(
        portfolio_turnover.index,
        portfolio_turnover.values,
        portfolio_turnover.mean(),
        color="gray",
        lw=1,
        label=u"turnover",
    )
    ax4.legend(loc="best")
    ax4.spines["top"].set_color("none")
    ax4.spines["right"].set_color("none")

    if fig_handler:
        plt.close()
        return fig

def portfolio_year_performance(
    portfolio_return, benchmark_return, portfolio_turnover_rate, freq='d'
):
    """
    计算多头分年收益
    :param portfolio_return:
    :param benchmark_return:
    :param portfolio_turnover_rate:
    :return:
    """
    return_free = cfg.PARAMS_BT["risk_free_rate"]  # 无风险利率
    d_const_1y = {'d': 252.0, 'w': 50.0, 'm': 12.0}
    const_1y = d_const_1y[freq]

    year = pd.Series(portfolio_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]

    # 年化性能
    perf_year = pd.DataFrame(
        np.nan,
        index=year,
        columns=[
            "YTD",
            "excess_return",
            "benchmark_return",
            "max_drawdown",
            "volatility",
            "sharp_ratio",
            "turnover_rate",
        ],
    )

    # 计算每年的性能
    for y in year:
        value = (portfolio_return[y] + 1).cumprod()  # 价值
        bch_value = (benchmark_return[y] + 1).cumprod()  # 基准价值

        max_down = 1 - min(value[y] / np.maximum.accumulate(value))  # 最大回撤
        volat = portfolio_return[y].std() * (const_1y ** 0.5)  # 组合波动率

        day_number = len(value[y])
        ret = value[-1] - 1  # 累计年收益
        bch_ret = bch_value[-1] - 1  # 基准指数收益率
        exs_ret = ret - bch_ret  # 超额收益率

        sharp = (ret - return_free) / volat  # 组合夏普值
        turnover = portfolio_turnover_rate[y].mean()  # 组合换手率
        perf_year.loc[y] = [ret, exs_ret, bch_ret, max_down, volat, sharp, turnover]
    return perf_year

def hedged_year_performance(hedged_return, benchmark_return, freq='d'):
    """
    计算对冲组合的分年表现
    :param hedged_return:
    :param benchmark_return:
    :return:
    """
    return_free = cfg.PARAMS_BT["risk_free_rate"]  # 无风险利率
    d_const_1y = {'d': 252.0, 'w': 50.0, 'm': 12.0}
    const_1y = d_const_1y[freq]

    year = pd.Series(hedged_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]

    # 组合性能
    perf_year = pd.DataFrame(
        np.nan,
        index=year,
        columns=[
            "YTD",
            "benchmark_return",
            "max_drawdown",
            "volatility",
            "sharp_ratio",
            "ret_draw_ratio",
        ],
    )

    # 计算每年的组合性能
    for y in year:
        # 对冲组合价值
        value = (hedged_return[y] + 1).cumprod()
        # 基准组合价值
        bch_value = (benchmark_return[y] + 1).cumprod()
        # 最大回撤
        max_down = 1 - min(value[y] / np.maximum.accumulate(value))
        # 波动率
        volat = hedged_return[y].std() * (const_1y ** 0.5)

        day_number = len(value[y])
        # 组合收益率
        ret = value[-1] - 1
        # 基准收益率
        bch_ret = bch_value[-1] - 1

        # 夏普比值
        sharp = (ret - return_free) / volat
        # 收益回撤比
        ret_draw_ratio = ret / max_down
        perf_year.loc[y] = [ret, bch_ret, max_down, volat, sharp, ret_draw_ratio]
    return perf_year

def format_year_performance(returns, benchmark_ind, turnover_rate):
    # 基准组合收益率
    benchmark_returns = benchmark_ind.pct_change()
    benchmark_returns[0] = 0

    # 计算组合年度性能
    perf_year = portfolio_year_performance(returns, benchmark_returns, turnover_rate)
    perf_year.columns = [u"累计收益", u"超额收益", u"基准收益", u"最大回撤", u"波动率", u"夏普比率", u"换手率"]
    perf_year.index.name = u"年份"
    format_funcs = {
        u"累计收益": "{:.2%}".format,
        u"超额收益": "{:.2%}".format,
        u"基准收益": "{:.2%}".format,
        u"最大回撤": "{:.2%}".format,
        u"波动率": "{:.2%}".format,
        u"夏普比率": "{:.2f}".format,
        u"换手率": "{:.3f}".format,
    }
    perf_year = perf_year.transform(format_funcs)
    return perf_year

def format_hedged_year_performance(returns, benchmark_ind):

    # 基准收益率
    benchmark_returns = benchmark_ind.pct_change()
    benchmark_returns[0] = 0
    perf_year = hedged_year_performance(returns, benchmark_returns)
    perf_year.columns = [u"累计收益", u"基准收益", u"最大回撤", u"波动率", u"夏普比率", u"收益回撤比"]
    perf_year.index.name = u"年份"
    format_funcs = {
        u"累计收益": "{:.2%}".format,
        u"基准收益": "{:.2%}".format,
        u"最大回撤": "{:.2%}".format,
        u"波动率": "{:.2%}".format,
        u"夏普比率": "{:.2f}".format,
        u"收益回撤比": "{:.3f}".format,
    }
    perf_year = perf_year.transform(format_funcs)
    return perf_year


class BackTest(object):
    """
    组合回测模块
    Parameters:
    -----------
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
    portfolios_create: 组合生成时间 格式'T' 或者 'T-1' 默认T-1 为 T时，应该注意选择交易价格在组合生成之后
    tday_lag 交易日延迟 default 1 设置为0时，表示当天交易， 配合portfolios_create使用
    trade_mode 交易模式 default: 'target_port' 调整到目标权重 'buy_diff' 卖出不在名单中的个股，买入新入选的个股，持有的个股保持仓位不变
    stock_pool_flag  Bool 是否有股票池的限制， 默认为 False, 当限制股票池是 设置为True
    stock_pool_aindex  list 指数成分股选股， 输入指数代码 eg.['000300.SH'] ['000300.SH', '000905.SH']
    stock_pool_industry  list 行业选股 输入SW1级行业代码 eg [6115000000000000, 6103000000000000]
    stock_pool_size list 按照流通市值由大到小取 [begin, end] eg[0, 1000] 取流通市值最大的1000

    Methods:
    --------
    run: 运行组合回测, 并统计和保存结果
            回测模块实例化后，run函数通过设置不同参数，可单独运行以获取不同区间的回
            测结果
        Parameters:
        -----------
        start_date: 回测起始日期, str, default None
            值为空时取实例化时的start参数的值，下同
        end_date: 回测结束日期, str, default None

        plot_options: 绘图选项, dict, default None
            目前可选键值有:
                对数y轴 "log_y": {None, "left", "right", or "both"}
                只画对冲曲线 "hedged_only": {True, False (default)}

    """
    def __init__(
            self,
            portfolios,
            account=10000000,
            start_date=None,
            end_date=None,
            fields_trade_prices_buy='stock_vwap_30m',
            bar_trade_prices_buy=1,
            fields_trade_prices_sell='stock_vwap_30m',
            bar_trade_prices_sell=1,
            portfolios_create='T-1',
            tday_lag=1,
            trade_mode='target_port',
            stock_pool_flag=False,
            stock_pool_aindex=[],
            stock_pool_industry=[],
            stock_pool_size=[],
    ):
        """  Args:
            portfolios: DataFrame
            start_date:str
            end_date:str
        """
        # add just start and end date
        portfolios.sort_index(inplace=True)
        self.portfolios = portfolios.copy()
        self._check_format_portfolios()
        self.accounts = account
        self.cash = account
        self.port_create_tday = portfolios_create
        self.tday_lag = tday_lag
        self.trade_mode = trade_mode
        if stock_pool_flag:
            self.trading_pool = True
            self.stock_pool_industry = stock_pool_industry
            self.stock_pool_aindex = stock_pool_aindex
            self.stock_pool_size = stock_pool_size
        else:
            self.trading_pool = False
        start_date_p = min(portfolios.index)
        end_date_p = max(portfolios.index)
        # 测试输入是否字符串，开始日期，结束日期

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 开始时间的设置
        if start_date is not None:
            self.start_date = start_date if start_date > start_date_p else start_date_p
        else:
            self.start_date = start_date_p

        # 结束时间的设置
        if end_date is not None:
            self.end_date = end_date if end_date < end_date_p else end_date_p
        else:
            self.end_date = end_date_p

        self._check_trday_data()  # 获取交易日历
        self._prepare_data(
            fields_trade_prices_buy,
            fields_trade_prices_sell,
            bar_trade_prices_buy,
            bar_trade_prices_sell
        )  # 获取历史数据

        self.tb_traderecords = []  # 交易记录
        self.tb_positions = []  # 持仓记录
        self.tb_accounts = []  # 账户记录
        self.port_hold = None  # 最新持仓

    def _check_format_portfolios(self):
        """检验输入组合的格式"""
        for col in ['code', 'weight', 'amt']:
            if col not in self.portfolios.columns:
                print('miss column %s' % (col))
                return

    def _check_trday_data(self):
        """交易日时间序列"""
        trdays = query_data.get_trade_days('d', exchange='SSE', from_trade_day=self.start_date, to_trade_day=self.end_date)
        trdays.sort()
        self.trdays = trdays.copy()
        self._tradedays = pd.to_datetime(trdays)

    def _prepare_data(self, fields_trade_prices_buy, fields_trade_prices_sell, bar_trade_prices_buy, bar_trade_prices_sell):
        """
        加载数据,读取行情数据
        """
        # 日行情
        tmp = fetch_data.fetch(self.start_date, self.end_date, cfg.fields_daily)
        self.eodprices = tmp.copy()
        codelst = self.eodprices['stock_close'].columns

        # 前收盘价，使用database数据填充缺失值
        sql = f"SELECT pre_close, maxup, maxdown, code, trade_date from ashareeodprices WHERE trade_date>='{self.start_date}' and trade_date<='{self.end_date}'"
        df_preclose = pd.read_sql_query(sql, con=query_data.PostgreSQL.get_engine('ftresearch'))
        tmpdf = pd.pivot_table(df_preclose, values='pre_close', index=['trade_date'], columns=['code'])
        columns = [v[:6] for v in tmpdf.columns]
        tmpdf.columns = columns
        tmpdf.index = pd.to_datetime(tmpdf.index)
        tmpdf.fillna(method='ffill', axis=0, inplace=True)
        self.eodprices['stock_lclose'].update(tmpdf)

        tmpdf = pd.pivot_table(df_preclose, values='maxup', index=['trade_date'], columns=['code'])
        columns = [v[:6] for v in tmpdf.columns]
        tmpdf.columns = columns
        tmpdf.index = pd.to_datetime(tmpdf.index)
        f_col_miss = [v for v in codelst if v not in tmpdf.columns]
        tmpdf.loc[:, f_col_miss] = np.nan
        self.eodprices['stock_maxup'] = tmpdf.copy()

        tmpdf = pd.pivot_table(df_preclose, values='maxdown', index=['trade_date'], columns=['code'])
        columns = [v[:6] for v in tmpdf.columns]
        tmpdf.columns = columns
        tmpdf.index = pd.to_datetime(tmpdf.index)
        f_col_miss = [v for v in codelst if v not in tmpdf.columns]
        tmpdf.loc[:, f_col_miss] = np.nan
        self.eodprices['stock_maxdown'] = tmpdf.copy()

        # 指数行情
        tmp = fetch_data.fetch(self.start_date, self.end_date, ['index_close', 'index_amount', 'index_open'])
        self.indexeodprices = tmp.copy()

        # 分钟行情,交易行情
        tmp_fields = [fields_trade_prices_buy]
        if fields_trade_prices_sell not in tmp_fields:
            tmp_fields.append(fields_trade_prices_sell)
        tmp = fetch_data.fetch(self.start_date, self.end_date, tmp_fields)
        if 'm' == fields_trade_prices_buy[-1]:
            minutes_buy = tmp[fields_trade_prices_buy].index[bar_trade_prices_buy-1].strftime('%H:%M:%S')
            f_tday_minutes_buy = pd.to_datetime(['%s %s' % (v, minutes_buy) for v in self.trdays])
            df_buyprices = tmp[fields_trade_prices_buy].loc[f_tday_minutes_buy, :]
            df_buyprices.index = self._tradedays
        else:
            df_buyprices = tmp[fields_trade_prices_buy].copy()

        if 'm' == fields_trade_prices_sell[-1]:
            minutes_sell = tmp[fields_trade_prices_sell].index[bar_trade_prices_sell-1].strftime('%H:%M:%S')
            f_tday_minutes_sell = pd.to_datetime(['%s %s' % (v, minutes_sell) for v in self.trdays])
            df_sellprices = tmp[fields_trade_prices_sell].loc[f_tday_minutes_sell, :]
            df_sellprices.index = self._tradedays
        else:
            df_sellprices = tmp[fields_trade_prices_sell]

        self._buyprices = df_buyprices.copy()
        self._sellprices = df_sellprices.copy()

        # 成分股权重
        # sql1 = f"select * from aindexconsfreeweight where trade_date>='{self.start_date}' and trade_date<='{self.end_date}'"
        # rawdf_table_aindexconsfreeweight = pd.read_sql_query(sql1, con=query_data.PostgreSQL.get_engine('ftresearch'))
        rawdf_table_aindexconsfreeweight = pd.read_sql_table('aindexconsfreeweight', con=query_data.PostgreSQL.get_engine('ftresearch'))
        rawdf_table_aindexconsfreeweight['trade_date'] = rawdf_table_aindexconsfreeweight['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        self.df_aindexcons = rawdf_table_aindexconsfreeweight.copy()
        tmp_trade_date_lst = list(rawdf_table_aindexconsfreeweight['trade_date'].unique())
        tmp_trade_date_lst.sort()
        self.aindexcons_trdate = tmp_trade_date_lst

        # 行业信息
        tmp = fetch_data.fetch(self.start_date, self.end_date, [cfg.col_industries_name])[cfg.col_industries_name]
        tmp.fillna(0, inplace=True)
        tmp = tmp.astype(int)
        self.df_industries = tmp.copy()


    def _prepare_tradingpool(self, trade_date, index_codelst=[], industry_codelst=[], size_bound=[]):
        """
        trade_date: 交易日 'yyyy-mm-dd'
        index_codelst: 指数成分股，从aindexconsfreeweight中获取 eg. index_codelst=['000905.SH', '000300.SH']
        industry_codelst: SW1 行业代码 int格式
        size_bound: list 按照流通市值过滤 [begin, end] 按照市值由大到小排序
        :return: 输出list 满足条件的股票列表
        """
        if type(index_codelst) == str:
            index_codelst = [index_codelst]
        if type(industry_codelst) == str:
            industry_codelst = [industry_codelst]
        newcodelst = []
        if len(index_codelst) > 0:
            f_trade_date = [v for v in self.aindexcons_trdate if v <= trade_date][-1]
            tmpdf = self.df_aindexcons[self.df_aindexcons['index_code'].isin(index_codelst)]
            min_tradedate = tmpdf['trade_date'].min()
            if f_trade_date<min_tradedate:
                print(f'trading date: {trade_date} ERROR! missing aindexcons data, use the {min_tradedate} replace')
                f_trade_date = min_tradedate
            f_codes = list(tmpdf.loc[tmpdf['trade_date']==f_trade_date, 'code'])
            f_codes = [v[:6] for v in f_codes]
            newcodelst.extend(f_codes)
        if len(industry_codelst)>0:
            tmpse=self.df_industries.loc[trade_date, :]
            f_codes = list(tmpse[tmpse.isin(industry_codelst)].index)
            if f_codes == []:
                print(f'trading date: {trade_date} ERROR! check the industry code SW1 int ')
            newcodelst.extend(f_codes)
        if len(size_bound) > 0:
            tmpse = self.eodprices['stock_mcap'].loc[trade_date, :]
            tmpse.sort_values(ascending=False, inplace=True)
            f_codes = list(tmpse.iloc[size_bound[0]: size_bound[1]].index)
            newcodelst.extend(f_codes)
        return newcodelst

    def get_dailyreturns(self, port_target=None, port_hold=None, tday_lag=1):
        """
        计算每天的收益率，计算时考虑涨跌停以及停牌
        :param port_hold:  组合持仓 default None 表示新策略，没有持仓，交易日为组合的下一个
        :param port_target:  组合目标持仓 default None 表示组合平仓
        :param tday_lag: 交易延迟，default 1 默认下一个交易日交易
        :return:
        """
        new_port_date = port_target.index[0].strftime('%Y-%m-%d')  # 目标组合的日期
        if tday_lag > 0:
            trade_date = query_data.next_tradingday(new_port_date, lag=tday_lag)  # 收益率区间结束日期
        else:
            trade_date = new_port_date
        buyprices = self._buyprices.loc[trade_date]
        codelst = list(buyprices.index)
        codelst.sort()
        buyprices = buyprices[codelst]
        sellprices = self._sellprices.loc[trade_date][codelst]
        uplimit = self.eodprices['stock_maxup'].loc[trade_date][codelst]
        dnlimit = self.eodprices['stock_maxdown'].loc[trade_date][codelst]
        closeprice = self.eodprices['stock_close'].loc[trade_date][codelst]
        pre_closeprice = self.eodprices['stock_lclose'].loc[trade_date][codelst]
        stock_suspend = self.eodprices['suspend'].loc[trade_date]
        buy_limited_AStocks = buyprices[buyprices > uplimit - 0.0001].index  # 过滤涨跌停股
        sell_limited_AStocks = sellprices[sellprices < dnlimit + 0.0001].index
        suspend_AStocks = stock_suspend[stock_suspend == 1].index

        f_cols_trade = ['trade_date', 'code', 'buy_amt', 'sell_amt', 'trade_price']
        f_cols_port = ['trade_date', 'code', 'amt']
        tmpdict_account = {'trade_date': trade_date}  # 当天的交易记录
        if (port_hold is not None) and (port_target is not None):  # 先卖后买，每期新的组合按照昨日的账户权益计算一个新的目标持仓
            tmpdf_hold = self.port_hold.copy()
            idx_notrade = tmpdf_hold['code'].isin(buy_limited_AStocks) | tmpdf_hold['code'].isin(
                suspend_AStocks) | tmpdf_hold['code'].isin(sell_limited_AStocks)
            df_hold = tmpdf_hold.loc[idx_notrade, :]
            df_sell = tmpdf_hold.loc[~idx_notrade, :]
            tmpdf_hold.set_index('code', inplace=True)

            # port_amt = self.accounts + self.cash  # 第二日的总市值是账户市值 + 现金资产
            port_amt_sell = df_sell['amt'].sum() + self.cash  # 第二日可交易的市值
            df_sell.set_index('code', inplace=True)
            trade_codelst = list(df_sell.index)

            tmpdf_port = port_target.copy()
            tmpdf_port.index = tmpdf_port['code']
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(
                suspend_AStocks) | tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port = tmpdf_port.loc[~idx_notrade, :]  # 排除持仓不能交易的股票
            tmpdf_port['adj_weight'] = tmpdf_port['weight'] / tmpdf_port['weight'].sum()
            tmpdf_port['amt'] = port_amt_sell * tmpdf_port['adj_weight']
            tmpdf_port.set_index('code', inplace=True)
            tmpdf_diff = pd.concat([df_sell['amt'], tmpdf_port['amt']], axis=1)
            tmpdf_diff.fillna(0, inplace=True)
            tmpdf_diff.columns = ['old_amt', 'new_amt']
            tmpdf_diff['diff'] = tmpdf_diff['new_amt'] - tmpdf_diff['old_amt']
            tmpdf_diff['hold'] = 0  # 初始化持有的仓位

            # 卖出
            idx_sell = tmpdf_diff['diff'] < 0
            tmpdf_diff.loc[idx_sell, 'hold'] = tmpdf_diff.loc[idx_sell, 'new_amt']  # 卖出剩下的市值为持有市值
            sellcodelsts = list(tmpdf_diff[idx_sell].index)
            tmpdf_port_sell = pd.DataFrame(index=sellcodelsts, columns=f_cols_trade)
            tmpdf_port_sell['trade_date'] = trade_date
            tmpdf_port_sell['sell_amt'] = -tmpdf_diff.loc[idx_sell, 'diff']
            tmpdf_port_sell['trade_price'] = sellprices[sellcodelsts]
            tmpdf_port_sell['code'] = sellcodelsts
            tmpdf_port_sell['buy_amt'] = 0
            tmpdf_port_sell['lclose'] = pre_closeprice[sellcodelsts]
            tmpdf_port_sell['pct_chg'] = 1.0*tmpdf_port_sell['trade_price']/tmpdf_port_sell['lclose']-1
            tmpdf_port_sell['pct_chg'].fillna(0, inplace=True)
            tmpdf_port_sell['sell_amt'] = tmpdf_port_sell['sell_amt']*(1+tmpdf_port_sell['pct_chg'])  # 当天卖出的金额是昨天的市值按照交易时刻的市值
            pnl_sell = (tmpdf_port_sell['sell_amt'] * tmpdf_port_sell['pct_chg'] / (1 + tmpdf_port_sell['pct_chg']) - \
                       tmpdf_port_sell['sell_amt'] * (cfg.PARAMS_BT['sell_commission']+cfg.PARAMS_BT['tax_ratio'])).sum()
            sell_amt = tmpdf_port_sell['sell_amt'].sum()+pnl_sell
            tmpdict_account['sell_amt'] = sell_amt
            self.cash += sell_amt
            self.tb_traderecords.append(tmpdf_port_sell.loc[:, f_cols_trade].copy())

            # 买入
            idx_buy = tmpdf_diff['diff'] > 0
            tmpdf_diff.loc[idx_buy, 'hold'] = tmpdf_diff.loc[idx_buy, 'new_amt'] - tmpdf_diff.loc[idx_buy, 'diff']  # 持有的市值
            buycodelsts = list(tmpdf_diff[idx_buy].index)
            tmpdf_port_buy = pd.DataFrame(index=buycodelsts, columns=f_cols_trade)
            tmpdf_port_buy['trade_date'] = trade_date
            tmpdf_port_buy['buy_amt'] = tmpdf_diff.loc[idx_buy, 'diff']
            tmpdf_port_buy['trade_price'] = buyprices[buycodelsts]
            tmpdf_port_buy['code'] = buycodelsts
            tmpdf_port_buy['sell_amt'] = 0
            tmpdf_port_buy['close'] = closeprice[buycodelsts]
            tmpdf_port_buy['pct_chg'] = 1.0 * tmpdf_port_buy['close'] / tmpdf_port_buy['trade_price'] - 1
            tmpdf_port_buy['pct_chg'].fillna(0, inplace=True)
            tmpdf_port_buy['amt'] = tmpdf_port_buy['buy_amt'] * (1+tmpdf_port_buy['pct_chg'])
            pnl_buy = (tmpdf_port_buy['buy_amt'] * tmpdf_port_buy['pct_chg'] - tmpdf_port_buy['buy_amt'] * cfg.PARAMS_BT['buy_commission']).sum()
            buy_amt = tmpdf_port_buy['buy_amt'].sum()
            tmpdict_account['buy_amt'] = buy_amt
            self.cash -= buy_amt*(1 + cfg.PARAMS_BT['buy_commission'])
            self.tb_traderecords.append(tmpdf_port_buy.loc[:, f_cols_trade].copy())

            # 持有的市值
            idx_hold = tmpdf_diff['hold'] > 0
            tmpdf_hold['hold_amt'] = tmpdf_hold['amt']
            codelst_hold = tmpdf_diff.loc[idx_hold, :].index
            tmpdf_hold.loc[codelst_hold, 'hold_amt'] = tmpdf_diff.loc[codelst_hold, 'hold']
            holdcodelsts = list(tmpdf_diff[idx_hold].index)
            holdcodelsts.extend(df_hold['code'].tolist())
            tmpdf_port_hold = pd.DataFrame(index=holdcodelsts, columns=f_cols_trade)
            tmpdf_port_hold['trade_date'] = trade_date
            tmpdf_port_hold['hold_amt'] = tmpdf_hold.loc[holdcodelsts, 'hold_amt']
            tmpdf_port_hold['close'] = closeprice[holdcodelsts]
            tmpdf_port_hold['code'] = holdcodelsts
            tmpdf_port_hold['lclose'] = pre_closeprice[holdcodelsts]
            tmpdf_port_hold['pct_chg'] = 1.0 * tmpdf_port_hold['close'] / tmpdf_port_hold['lclose'] - 1
            tmpdf_port_hold['pct_chg'].fillna(0, inplace=True)
            tmpdf_port_hold['amt'] = tmpdf_port_hold['hold_amt']*(1+tmpdf_port_hold['pct_chg'])
            pnl_hold = (tmpdf_port_hold['hold_amt'] * tmpdf_port_hold['pct_chg']).sum()

            pnl = pnl_hold + pnl_buy + pnl_sell
            tmpdict_account['pnl'] = pnl
            tmpdict_account['pnl_buy'] = pnl_buy
            tmpdict_account['pnl_sell'] = pnl_sell
            tmpdict_account['turnover'] = sell_amt / self.accounts
            self.accounts += pnl
            tmpdict_account['cash'] = self.cash.copy()
            tmpdict_account['balance'] = self.accounts.copy()
            self.tb_accounts.append(tmpdict_account.copy())

            # 持仓：未交易+目标
            tmpdf = pd.concat([tmpdf_port_hold['amt'], tmpdf_port_buy['amt']], axis=1)
            tmpdf.fillna(0, inplace=True)
            tmpdf.columns = ['amt_hold', 'amt_buy']
            tmpdf['amt'] = tmpdf['amt_hold'] + tmpdf['amt_buy']
            tmpdf['trade_date'] = trade_date
            tmpdf['code'] = tmpdf.index
            self.tb_positions.append(tmpdf.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf.loc[:, f_cols_port].copy()
            return tmpdf.loc[:, f_cols_port].copy()
        if (port_hold is None) and (port_target is not None):
            # 新开仓只买入, 将涨跌停，停牌的股票权重设置成0
            tmpdf_port = port_target.copy()
            tmpdf_port.index = tmpdf_port['code']
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(suspend_AStocks) | \
                          tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port.loc[idx_notrade, 'weight'] = 0
            buycodelst = tmpdf_port.loc[~idx_notrade, 'code'].values
            tmpdf_port = tmpdf_port.loc[buycodelst, :]  # 开仓只对可以买入de建仓
            tmpdf_port['adj_weight'] = tmpdf_port['weight'] / tmpdf_port['weight'].sum()
            tmpdf_port.loc[buycodelst, 'trade_price'] = buyprices[buycodelst]
            buy_amt = self.cash * (1-cfg.PARAMS_BT['buy_commission'])
            tmpdf_port['buy_amt'] = tmpdf_port['adj_weight'] * buy_amt
            tmpdf_port['sell_amt'] = 0
            tmpdf_port['volume'] = round(tmpdf_port['buy_amt'] / tmpdf_port['trade_price']/100.0, 0)*100
            self.cash -= (tmpdf_port['volume'] * tmpdf_port['trade_price']*(1+cfg.PARAMS_BT['buy_commission'])).sum()
            tmpdict_account['buy_amt'] = (tmpdf_port['volume'] * tmpdf_port['trade_price']).sum()
            tmpdict_account['sell_amt'] = 0
            tmpdict_account['cash'] = self.cash.copy()

            # 计算当天收益
            tmpdf_port.loc[buycodelst, 'close'] = closeprice[buycodelst]
            tmpdf_port['amt'] = tmpdf_port['close'] * tmpdf_port['volume']
            pnl = ((tmpdf_port['close'] - tmpdf_port['trade_price']*(1+cfg.PARAMS_BT['buy_commission']))*tmpdf_port['volume']).sum()
            tmpdict_account['pnl'] = pnl
            tmpdict_account['pnl_buy'] = pnl
            tmpdict_account['pnl_sell'] = 0
            tmpdict_account['turnover'] = np.nan
            self.accounts += pnl
            tmpdict_account['balance'] = self.accounts.copy()
            tmpdf_port['trade_date'] = trade_date
            self.tb_accounts.append(tmpdict_account.copy())
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())
            self.tb_positions.append(tmpdf_port.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf_port.loc[:, f_cols_port].copy()
            return tmpdf_port.loc[:, f_cols_port].copy()
        if (port_hold is not None) and (port_target is None):  # 平仓
            tmpdf_port = port_hold.copy()
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(suspend_AStocks) | \
                          tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port_notrade = tmpdf_port.loc[idx_notrade, :]
            tmpdf_port = tmpdf_port.loc[~idx_notrade, :]
            tmpdf_port['trade_price'] = sellprices[tmpdf_port['code']]
            tmpdf_port['buy_amt'] = 0
            tmpdf_port['sell_amt'] = tmpdf_port['amt']
            tmpdf_port['lclose'] = pre_closeprice[tmpdf_port['code']]
            tmpdf_port['pct_chg'] = 1.0 * tmpdf_port['trade_price'] / tmpdf_port['lclose'] - 1
            tmpdf_port['pct_chg'].fillna(0, inplace=True)
            tmpdf_port['sell_amt'] = tmpdf_port['sell_amt'] * (
                        1 + tmpdf_port['pct_chg'])  # 当天卖出的金额是昨天的市值按照交易时刻的市值
            pnl_sell = (tmpdf_port['sell_amt'] * tmpdf_port['pct_chg'] / (1 + tmpdf_port['pct_chg']) - \
                        tmpdf_port['sell_amt'] * (cfg.PARAMS_BT['sell_commission']+cfg.PARAMS_BT['tax_ratio'])).sum()
            sell_amt = tmpdf_port['sell_amt'].sum()+pnl_sell
            tmpdict_account['sell_amt'] = sell_amt
            self.cash += sell_amt
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())

            # 计算当天收益
            tmpdict_account['pnl'] = pnl_sell
            tmpdict_account['pnl_buy'] = 0
            tmpdict_account['pnl_sell'] = pnl_sell
            tmpdict_account['turnover'] = np.nan
            self.accounts += pnl_sell
            tmpdict_account['balance'] = self.accounts.copy()
            tmpdf_port['trade_date'] = trade_date
            self.tb_accounts.append(tmpdict_account.copy())
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())
            self.tb_positions.append(tmpdf_port_notrade.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf_port_notrade.loc[:, f_cols_port].copy()
            return tmpdf_port_notrade.loc[:, f_cols_port].copy()
        if (port_hold is None) and (port_target is None):
            return print('no data! Check the input data')

    def get_dailyreturns_buydiff(self, port_target=None, port_hold=None, tday_lag=1):
        """
        计算每天的收益率，计算时考虑涨跌停以及停牌, 不调整持仓股票权重，卖出不持仓股票
        :param port_hold:  组合持仓 default None 表示新策略，没有持仓，交易日为组合的下一个
        :param port_target:  组合目标持仓 default None 表示组合平仓
        :param tday_lag: 交易延迟，default 1 默认下一个交易日交易
        :return:
        """
        new_port_date = port_target.index[0].strftime('%Y-%m-%d')  # 目标组合的日期
        if tday_lag > 0:
            trade_date = query_data.next_tradingday(new_port_date, lag=tday_lag)  # 收益率区间结束日期
        else:
            trade_date = new_port_date
        buyprices = self._buyprices.loc[trade_date]
        codelst = list(buyprices.index)
        codelst.sort()
        buyprices = buyprices[codelst]
        sellprices = self._sellprices.loc[trade_date][codelst]
        uplimit = self.eodprices['stock_maxup'].loc[trade_date][codelst]
        dnlimit = self.eodprices['stock_maxdown'].loc[trade_date][codelst]
        closeprice = self.eodprices['stock_close'].loc[trade_date][codelst]
        pre_closeprice = self.eodprices['stock_lclose'].loc[trade_date][codelst]
        stock_suspend = self.eodprices['suspend'].loc[trade_date]
        buy_limited_AStocks = buyprices[buyprices > uplimit - 0.0001].index  # 过滤涨跌停股
        sell_limited_AStocks = sellprices[sellprices < dnlimit + 0.0001].index
        suspend_AStocks = stock_suspend[stock_suspend == 1].index

        f_cols_trade = ['trade_date', 'code', 'buy_amt', 'sell_amt', 'trade_price']
        f_cols_port = ['trade_date', 'code', 'amt']
        tmpdict_account = {'trade_date': trade_date}  # 当天的交易记录
        if (port_hold is not None) and (port_target is not None):  # 先卖后买，每期新的组合按照昨日的账户权益计算一个新的目标持仓
            tmpdf_hold = self.port_hold.copy()
            idx_notrade = tmpdf_hold['code'].isin(buy_limited_AStocks) | tmpdf_hold['code'].isin(
                suspend_AStocks) | tmpdf_hold['code'].isin(sell_limited_AStocks)
            df_hold = tmpdf_hold.loc[idx_notrade, :]
            df_sell = tmpdf_hold.loc[~idx_notrade, :]
            tmpdf_hold.set_index('code', inplace=True)

            df_sell.set_index('code', inplace=True)
            trade_codelst = list(df_sell.index)

            tmpdf_port = port_target.copy()
            tmpdf_port.index = tmpdf_port['code']
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(
                suspend_AStocks) | tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port = tmpdf_port.loc[~idx_notrade, :]  # 排除持仓不能交易的股票
            target_port_codelst = list(tmpdf_port.index)
            sellcodelsts = [v for v in trade_codelst if v not in target_port_codelst]
            pnl_buy = 0
            pnl_sell = 0
            sell_amt = 0
            buycodelsts = []
            if len(sellcodelsts) > 0:
                # 卖出
                tmpdf_port_sell = pd.DataFrame(index=sellcodelsts, columns=f_cols_trade)
                tmpdf_port_sell['trade_date'] = trade_date
                tmpdf_port_sell['sell_amt'] = df_sell.loc[sellcodelsts, 'amt']
                tmpdf_port_sell['trade_price'] = sellprices[sellcodelsts]
                tmpdf_port_sell['code'] = sellcodelsts
                tmpdf_port_sell['buy_amt'] = 0
                tmpdf_port_sell['lclose'] = pre_closeprice[sellcodelsts]
                tmpdf_port_sell['pct_chg'] = 1.0*tmpdf_port_sell['trade_price']/tmpdf_port_sell['lclose']-1
                tmpdf_port_sell['pct_chg'].fillna(0, inplace=True)
                tmpdf_port_sell['sell_amt'] = tmpdf_port_sell['sell_amt'] * (
                            1 + tmpdf_port_sell['pct_chg'])  # 当天卖出的金额是昨天的市值按照交易时刻的市值
                pnl_sell = (tmpdf_port_sell['sell_amt'] * tmpdf_port_sell['pct_chg'] / (1 + tmpdf_port_sell['pct_chg']) - \
                           tmpdf_port_sell['sell_amt'] * (cfg.PARAMS_BT['sell_commission']+cfg.PARAMS_BT['tax_ratio'])).sum()
                sell_amt = tmpdf_port_sell['sell_amt'].sum()+pnl_sell
                tmpdict_account['sell_amt'] = sell_amt
                self.cash += sell_amt
                self.tb_traderecords.append(tmpdf_port_sell.loc[:, f_cols_trade].copy())

                # 买入
                buycodelsts = [v for v in target_port_codelst if v not in trade_codelst]
                if len(buycodelsts) > 0:
                    tmp_target_port = tmpdf_port.loc[buycodelsts, :]
                    tmp_target_port['adj_weight'] = tmp_target_port['weight'] / tmp_target_port['weight'].sum()
                    tmp_target_port['buy_amt'] = tmp_target_port['adj_weight'] * sell_amt
                    tmpdf_port_buy = pd.DataFrame(index=buycodelsts, columns=f_cols_trade)
                    tmpdf_port_buy['trade_date'] = trade_date
                    tmpdf_port_buy['buy_amt'] = tmp_target_port.loc[buycodelsts, 'buy_amt']
                    tmpdf_port_buy['trade_price'] = buyprices[buycodelsts]
                    tmpdf_port_buy['code'] = buycodelsts
                    tmpdf_port_buy['sell_amt'] = 0
                    tmpdf_port_buy['close'] = closeprice[buycodelsts]
                    tmpdf_port_buy['pct_chg'] = 1.0 * tmpdf_port_buy['close'] / tmpdf_port_buy['trade_price'] - 1
                    tmpdf_port_buy['pct_chg'].fillna(0, inplace=True)
                    tmpdf_port_buy['amt'] = tmpdf_port_buy['buy_amt'] * (1+tmpdf_port_buy['pct_chg'])
                    pnl_buy = (tmpdf_port_buy['buy_amt'] * tmpdf_port_buy['pct_chg'] - tmpdf_port_buy['buy_amt'] * cfg.PARAMS_BT['buy_commission']).sum()
                    buy_amt = tmpdf_port_buy['buy_amt'].sum()
                    tmpdict_account['buy_amt'] = buy_amt
                    self.cash -= buy_amt*(1 + cfg.PARAMS_BT['buy_commission'])
                    self.tb_traderecords.append(tmpdf_port_buy.loc[:, f_cols_trade].copy())

            # 持有的市值
            holdcodelsts = [v for v in tmpdf_hold.index if v in port_target['code'].tolist()]
            for tmpcode in df_hold['code'].tolist():
                if tmpcode not in holdcodelsts:
                    holdcodelsts.append(tmpcode)
            if len(holdcodelsts) > 0:
                tmpdf_port_hold = pd.DataFrame(index=holdcodelsts, columns=f_cols_trade)
                tmpdf_port_hold['trade_date'] = trade_date
                tmpdf_port_hold['hold_amt'] = tmpdf_hold.loc[holdcodelsts, 'amt']
                tmpdf_port_hold['close'] = closeprice[holdcodelsts]
                tmpdf_port_hold['code'] = holdcodelsts
                tmpdf_port_hold['lclose'] = pre_closeprice[holdcodelsts]
                tmpdf_port_hold['pct_chg'] = 1.0 * tmpdf_port_hold['close'] / tmpdf_port_hold['lclose'] - 1
                tmpdf_port_hold['pct_chg'].fillna(0, inplace=True)
                tmpdf_port_hold['amt'] = tmpdf_port_hold['hold_amt']*(1+tmpdf_port_hold['pct_chg'])
                pnl_hold = (tmpdf_port_hold['hold_amt'] * tmpdf_port_hold['pct_chg']).sum()

                pnl = pnl_hold + pnl_buy + pnl_sell
                tmpdict_account['pnl'] = pnl
                tmpdict_account['pnl_buy'] = pnl_buy
                tmpdict_account['pnl_sell'] = pnl_sell
                tmpdict_account['turnover'] = sell_amt / self.accounts
                self.accounts += pnl
                tmpdict_account['cash'] = self.cash.copy()
                tmpdict_account['balance'] = self.accounts.copy()
                self.tb_accounts.append(tmpdict_account.copy())

            # 持仓：未交易+目标
            if len(holdcodelsts)>0 and len(buycodelsts)>0:
                tmpdf = pd.concat([tmpdf_port_hold['amt'], tmpdf_port_buy['amt']], axis=1)
                tmpdf.fillna(0, inplace=True)
                tmpdf.columns = ['amt_hold', 'amt_buy']
                tmpdf['amt'] = tmpdf['amt_hold'] + tmpdf['amt_buy']
            elif len(holdcodelsts)>0:
                tmpdf = tmpdf_port_hold.copy()
            elif len(buycodelsts)>0:
                tmpdf = tmpdf_port_buy.copy()
            else:
                print('Error no Trade')
            tmpdf['trade_date'] = trade_date
            tmpdf['code'] = tmpdf.index
            self.tb_positions.append(tmpdf.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf.loc[:, f_cols_port].copy()
            return tmpdf.loc[:, f_cols_port].copy()
        if (port_hold is None) and (port_target is not None):
            # 新开仓只买入, 将涨跌停，停牌的股票权重设置成0
            tmpdf_port = port_target.copy()
            tmpdf_port.index = tmpdf_port['code']
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(suspend_AStocks) | \
                          tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port.loc[idx_notrade, 'weight'] = 0
            buycodelst = tmpdf_port.loc[~idx_notrade, 'code'].values
            tmpdf_port = tmpdf_port.loc[buycodelst, :]  # 开仓只对可以买入de建仓
            tmpdf_port['adj_weight'] = tmpdf_port['weight'] / tmpdf_port['weight'].sum()
            tmpdf_port.loc[buycodelst, 'trade_price'] = buyprices[buycodelst]
            buy_amt = self.cash * (1-cfg.PARAMS_BT['buy_commission'])
            tmpdf_port['buy_amt'] = tmpdf_port['adj_weight'] * buy_amt
            tmpdf_port['sell_amt'] = 0
            tmpdf_port['volume'] = round(tmpdf_port['buy_amt'] / tmpdf_port['trade_price']/100.0, 0)*100
            self.cash -= (tmpdf_port['volume'] * tmpdf_port['trade_price']*(1+cfg.PARAMS_BT['buy_commission'])).sum()
            tmpdict_account['buy_amt'] = (tmpdf_port['volume'] * tmpdf_port['trade_price']).sum()
            tmpdict_account['sell_amt'] = 0
            tmpdict_account['cash'] = self.cash.copy()

            # 计算当天收益
            tmpdf_port.loc[buycodelst, 'close'] = closeprice[buycodelst]
            tmpdf_port['amt'] = tmpdf_port['close'] * tmpdf_port['volume']
            pnl = ((tmpdf_port['close'] - tmpdf_port['trade_price']*(1+cfg.PARAMS_BT['buy_commission']))*tmpdf_port['volume']).sum()
            tmpdict_account['pnl'] = pnl
            tmpdict_account['pnl_buy'] = pnl
            tmpdict_account['pnl_sell'] = 0
            tmpdict_account['turnover'] = np.nan
            self.accounts += pnl
            tmpdict_account['balance'] = self.accounts.copy()
            tmpdf_port['trade_date'] = trade_date
            self.tb_accounts.append(tmpdict_account.copy())
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())
            self.tb_positions.append(tmpdf_port.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf_port.loc[:, f_cols_port].copy()
            return tmpdf_port.loc[:, f_cols_port].copy()
        if (port_hold is not None) and (port_target is None):  # 平仓
            tmpdf_port = port_hold.copy()
            idx_notrade = tmpdf_port['code'].isin(buy_limited_AStocks) | tmpdf_port['code'].isin(suspend_AStocks) | \
                          tmpdf_port['code'].isin(sell_limited_AStocks)
            tmpdf_port_notrade = tmpdf_port.loc[idx_notrade, :]
            tmpdf_port = tmpdf_port.loc[~idx_notrade, :]
            tmpdf_port['trade_price'] = sellprices[tmpdf_port['code']]
            tmpdf_port['buy_amt'] = 0
            tmpdf_port['sell_amt'] = tmpdf_port['amt']
            tmpdf_port['lclose'] = pre_closeprice[tmpdf_port['code']]
            tmpdf_port['pct_chg'] = 1.0 * tmpdf_port['trade_price'] / tmpdf_port['lclose'] - 1
            tmpdf_port['pct_chg'].fillna(0, inplace=True)
            tmpdf_port['sell_amt'] = tmpdf_port['sell_amt'] * (
                        1 + tmpdf_port['pct_chg'])  # 当天卖出的金额是昨天的市值按照交易时刻的市值
            pnl_sell = (tmpdf_port['sell_amt'] * tmpdf_port['pct_chg'] / (1 + tmpdf_port['pct_chg']) - \
                        tmpdf_port['sell_amt'] * (cfg.PARAMS_BT['sell_commission']+cfg.PARAMS_BT['tax_ratio'])).sum()
            sell_amt = tmpdf_port['sell_amt'].sum()+pnl_sell
            tmpdict_account['sell_amt'] = sell_amt
            self.cash += sell_amt
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())

            # 计算当天收益
            tmpdict_account['pnl'] = pnl_sell
            tmpdict_account['pnl_buy'] = 0
            tmpdict_account['pnl_sell'] = pnl_sell
            tmpdict_account['turnover'] = np.nan
            self.accounts += pnl_sell
            tmpdict_account['balance'] = self.accounts.copy()
            tmpdf_port['trade_date'] = trade_date
            self.tb_accounts.append(tmpdict_account.copy())
            self.tb_traderecords.append(tmpdf_port.loc[:, f_cols_trade].copy())
            self.tb_positions.append(tmpdf_port_notrade.loc[:, f_cols_port].copy())
            self.port_hold = tmpdf_port_notrade.loc[:, f_cols_port].copy()
            return tmpdf_port_notrade.loc[:, f_cols_port].copy()
        if (port_hold is None) and (port_target is None):
            return print('no data! Check the input data')

    def backtest(self, start_date, end_date):
        """
        按照交易日滚动计算，此处进行股票池的约束
        :param start_date:  回测开始日期
        :param end_date:  回测截止日期
        :param tday_lag:  交易延迟 default1
        :return:
        """
        # 将回测日期队列写入日期参数
        bt_tdays = self.trdays

        f_bt_tdays = [v for v in bt_tdays if v>start_date and v<=end_date]
        #        global AStocks_buy_price, day, shr_prc, w_optimal, w_lastday
        last_hold_port = None  # 昨日持仓
        target_hold_port = None  # 目标持仓
        if self.tday_lag>0:
            f_bt_tdays = f_bt_tdays[:-self.tday_lag]
        for i, day in enumerate(f_bt_tdays):  #
            print("tday: {}".format(day))  # 这里改为不打印
            if self.port_create_tday == 'T':
                date_port = day
            else:
                date_port = query_data.prev_tradingday(day)  # 组合生成日期
            target_hold_port = self.portfolios.loc[date_port, :]  # 检验目标组合是否合格
            # 股票池约束
            if self.trading_pool:
                codelst = self._prepare_tradingpool(date_port, index_codelst=self.stock_pool_aindex,
                                                    industry_codelst=self.stock_pool_industry,
                                                    size_bound=self.stock_pool_size)
                target_hold_port = target_hold_port[target_hold_port['code'].isin(codelst)]
            if len(target_hold_port)==0:
                print('ERROR! no stock in pool')
                break
            if self.trade_mode=='target_port':
                last_hold_port = self.get_dailyreturns(port_target=target_hold_port, port_hold=last_hold_port, tday_lag=self.tday_lag)
            elif self.trade_mode=='buy_diff':
                last_hold_port = self.get_dailyreturns_buydiff(port_target=target_hold_port, port_hold=last_hold_port,
                                                   tday_lag=self.tday_lag)
            else:
                print('请确定交易方式')


    def performance(self, start_date, end_date, plot_options=None, freq='d', image_title='nav'):
        """ 收益统计
        :param start_date 开始日期
        :param end_date 截止日期
        :param plot_options: 绘图选项
        :param freq: 统计频率
        """
        df_accounts = pd.DataFrame(self.tb_accounts)
        df_accounts['ret'] = df_accounts['balance'].pct_change()
        df_accounts['ret'].fillna(0, inplace=True)
        df_accounts.index = pd.to_datetime(df_accounts['trade_date'])
        df_accounts.sort_index(inplace=True)

        trade_returns_series = df_accounts['ret'].loc[start_date: end_date]
        turnover_series = df_accounts['turnover'].loc[start_date: end_date]


        benchmark = self.indexeodprices['index_close'][cfg.PARAMS_BT['benchmark']]  # 基准指数收盘价
        benchmark.sort_index(inplace=True)
        benchmark_ret = benchmark.pct_change()
        benchmark_ret.fillna(0, inplace=True)
        benchmark = benchmark.loc[start_date:end_date]
        benchmark_ret = benchmark_ret.loc[start_date:end_date]

        # 调整数据频率
        if not freq == 'd':
            trade_returns_series = change_freq(trade_returns_series.copy(), mode='ret', to_freq=freq)
            turnover_series = change_freq(turnover_series.copy(), mode='turnover', to_freq=freq)
            benchmark = change_freq(benchmark.copy(), mode='nav', to_freq=freq)
            benchmark_ret = change_freq(benchmark.copy(), mode='ret',to_freq=freq)

        perf = portfolio_performance(trade_returns_series, benchmark, cfg.PARAMS_BT["risk_free_rate"])

        perf_yearly = format_year_performance(
            trade_returns_series, benchmark, turnover_series
        )
        hedged_returns = trade_returns_series - benchmark_ret
        hedged_returns.dropna(inplace=True)
        hedged_perf = portfolio_performance(
            hedged_returns, benchmark, cfg.PARAMS_BT["risk_free_rate"]
        )
        hedged_perf_yearly = format_hedged_year_performance(hedged_returns, benchmark)

        if plot_options is None:
            plot_options = {}
        self.image_name = image_title
        self.fig = plot_portfolio_performance(
            trade_returns_series,
            turnover_series,
            hedged_returns,
            benchmark,
            perf,
            hedged_perf,
            image_title,
            fig_handler=True,
            **plot_options
        )

        self.results = OrderedDict(
            {
                "trade_returns": trade_returns_series,
                "turnover_series": turnover_series,
                "positions_record": self.tb_positions,
                "trade_records": self.tb_traderecords,
                "accounts_records": self.tb_accounts,
                "hedged_returns": hedged_returns,
                "hedged_perf": hedged_perf,
                "perf": perf,
                "perf_yearly": perf_yearly,
                "hedged_perf_yearly": hedged_perf_yearly,
            }
        )

    def save_result(self, start_date, end_date):
        #
        self.fig.savefig(
            os.path.join(self.filepath, self.image_name + "_%s-%s.png" % (start_date, end_date))
        )
        with open(os.path.join(self.filepath, "%s_result_%s-%s.pkl" % (self.image_name, start_date, end_date)),
                  "wb", ) as f:
            pickle.dump(self.results, f, -1)

        path_ret = os.path.join(self.filepath, "%s_returns_%s-%s.csv" % (self.image_name, start_date, end_date))
        self.results['hedged_returns'].to_csv(path_ret, encoding='gb18030')

        path_account = os.path.join(self.filepath, "%s_accounts_%s-%s.csv" % (self.image_name, start_date, end_date))
        pd.DataFrame(self.results['accounts_records']).to_csv(path_account, encoding='gb18030')



    def run(self, start_date=None, end_date=None, plot_options=None, filename='port_20201221'):
        self.filepath = os.path.join(cfg.path_results, filename)
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        if start_date is None:
            start_date = self.start_date.strftime('%Y-%m-%d')

        if end_date is None:
            end_date = self.end_date.strftime('%Y-%m-%d')

        if not query_data.is_tradingday(start_date):
            start_date = query_data.next_tradingday(start_date)
        if not query_data.is_tradingday(end_date):
            end_date = query_data.prev_tradingday(end_date)

        # 启动任务
        self.backtest(start_date, end_date)
        self.performance(start_date, end_date, plot_options)
        self.save_result(start_date, end_date)

def get_portfolios():
    """
    获取测试组合
    :return:
    """
    # 情形1
    # df_port = pd.read_excel('..\\src\\results\\portfolio_management\\port_wangfeng.xlsx', index_col=0)  # wangfeng02
    # df_port['code'] = df_port['code'].astype('str').str.zfill(6)
    # df_port.index = pd.to_datetime(df_port.index)
    # df = df_port.copy()
    # 情形2
    # df_port = pd.read_csv('..\\src\\results\\portfolio_management\\ports_20201230_nosize.csv', index_col=0)  # yyshi
    df_port = pd.read_csv('..\\src\\results\\portfolio_management\\data_weight.csv', index_col=0)  # wangfeng01
    df_port.columns = [str(int(v)).zfill(6) for v in df_port.columns]
    df_port.index = pd.to_datetime(df_port.index)
    df = df_port.stack().reset_index()
    df.columns = ['trade_date', 'code', 'weight']
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    init_account = 10000000.0
    df['amt'] = init_account*df['weight']
    return df.copy()

def test_returns_20201225():
    """
    调试程序
    :return:
    """
    portfolios = get_portfolios()
    pro_bt = BackTest(portfolios, start_date='2020-02-01', end_date='2020-07-31')
    pro_bt.run(start_date='2020-02-01', end_date='2020-07-31', filename='test_20201230_wangfeng_weight')

def test_return_20210106():
    df_port = pd.read_csv('..\\src\\results\\portfolio_management\\weight_models.csv', index_col=0)  # 由模型训练出来的初始权重
    df_port.columns = [str(int(v)).zfill(6) for v in df_port.columns]
    df_port.index = pd.to_datetime(df_port.index)
    df = df_port.stack().reset_index()
    df.columns = ['trade_date', 'code', 'weight']
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    init_account = 10000000.0
    df['amt'] = init_account * df['weight']
    df = df[df['amt']>0]
    portfolios = df.copy()
    pro_bt = BackTest(portfolios, start_date='2017-10-26', end_date='2020-06-10', fields_trade_prices_buy='stock_close',
            fields_trade_prices_sell='stock_close', portfolios_create='T', tday_lag=1, trade_mode='target_port')
    pro_bt.run(start_date='2017-10-27', end_date='2020-06-10', filename='test_20210106_2017_check_sellall_T0_T1close')

def test_return_20210106_v1():
    df_port = pd.read_csv(r'E:\Share\Alpha\YYShi\backtests\\cshu_v0_trade.csv', index_col=0)  # wangfeng01
    df_port.rename(columns={'SYMBOL': 'code', 'date': 'trade_date', 'alpha_new': 'weight'}, inplace=True)
    # df_port['weight'] = 1/200.0
    df_port['code'] = df_port['code'].astype('str').str.zfill(6)
    df_port.set_index('trade_date', inplace=True)
    df_port.index = pd.to_datetime(df_port.index)
    df = df_port.copy()
    init_account = 10000000.0
    df['amt'] = init_account * df['weight']
    portfolios = df.copy()
    pro_bt = BackTest(portfolios, start_date='2017-04-05', end_date='2019-12-30')
    pro_bt.run(start_date='2017-04-05', end_date='2019-12-30', filename='test_20210108_cshu_v0_equalweight')

def test_return_20210107():
    df_port = pd.read_csv('..\\src\\results\\portfolio_management\\price_stocknum.csv', index_col=0)  # wangfeng01
    df_port.columns = [str(int(v)).zfill(6) for v in df_port.columns]
    df_port.index = pd.to_datetime(df_port.index)
    df_port = (df_port.T / df_port.T.sum()).T
    df = df_port.stack().reset_index()
    df.columns = ['trade_date', 'code', 'weight']
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    init_account = 10000000.0
    df['amt'] = init_account * df['weight']
    df = df[df['amt']>0]
    df.to_csv(r'E:\YuanyuanShi\Python\Alpha\src\results\portfolio_management\stocknum.csv')
    portfolios = df.copy()
    pro_bt = BackTest(portfolios, start_date='2017-10-26', end_date='2020-06-10', fields_trade_prices_buy='stock_open',
                      fields_trade_prices_sell='stock_open', stock_pool_flag=False, stock_pool_size=[0, 2000])
    pro_bt.run(start_date='2017-10-27', end_date='2020-06-10', filename='test_20210107_2017_check_sizetop2000_v6')


if __name__ == '__main__':
    # test_returns_20201225()
    # test_return_20210106()
    # test_return_20210106_v1()
    test_return_20210107()
