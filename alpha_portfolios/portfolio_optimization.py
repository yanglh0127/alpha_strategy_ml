# python3

"""
主要实现组合优化，得到满足约束条件的最优组合
本次主要功能为参照王峰使用的matlab程序，基于python实现(基于杨峰的PM)
@author: yyshi
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from cvxpy.error import DCPError, SolverError
import time
import os

from ft_platform.factor_process.fetch import fetch_factor
from ft_platform import fetch_data
from utils_func.utils import *
from utils_func import query_data
from alpha_portfolios import config as cfg

def get_target_data(start_date='1991-01-01', end_date='2099-12-31'):
    """
    获取优化的目标值，个股收益率或者是个股得分，需要考虑剔除ST，新股，黑名单的个股等情况
    columns = ['trade_date', 'code', 'score']
    :param start_date: str 'yyyy-mm-dd'  开始时间
    :param end_date: str 'yyyy-mm-dd' 截止时间
    :return:
    """
    localpath = cfg.path_data_asharescore
    df = pd.read_csv(localpath)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.loc[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date), :]
    df['code'] = df['stock_code'].astype('str').str.zfill(6)
    # df.loc[df['stock_code'] < 600000, 'code'] = df.loc[df['stock_code'] < 600000, 'code'] + '.SZ'
    # df.loc[~(df['stock_code'] < 600000), 'code'] = df.loc[~(df['stock_code'] < 600000), 'code'] + '.SH'
    df['score'] = df['y_hat']  # 个股得分取预测值
    f_cols = ['trade_date', 'code', 'score']
    newdf = pd.pivot_table(df, values='score', index=['trade_date'], columns=['code'])
    return newdf

class Portfolio_Management(object):
    """
    组合优化
    Paramerer:
    -----------
    df_target: pd.DataFrame  个股打分，如果是收益率，则要考虑交易日的移位
    start_date: str 起始日期
    end_date: str 截止日期
    num_holdings: int default 100 目标持股数量，只有当实际持股数超过目标持股数一定范围才起作用
    industry_neutral:  行业中性化, str, list or bool, default None
            只中性化单个或多个行业时，可输入包含这些行业名称的列表；输入True时，对所有行业进行中性化
    style_neutral: 风格中性化 str, list or bool, default False 不做风格约束 输入True时，对所有风格进行中性化
    style_expourse: 风格暴露 str, list or bool, default None, 在配置文件中有风格暴露的参数
    Epsilon: 中性化约束阈值
    st_or_not: 是否包含ST的个股，bool default False 剔除ST
    upper_bound: 个股权重上限, float, default None
    lower_bound: 个股权重下限, float, default None
    to_bound: 换手率约束, float, default None
    te_bound: 跟踪误差约束, float, default None
            注意，须有因子和个股风险的数据，否则会报错，下同
    vol_bound: 组合波动约束, float, default None
    signal_direcation: 个股打分与预期收益的相关性方向, int, default 1
            负向（打分越低，预期收益越高）时输入-1
    disp_info: 优化过程中是否显示信息, str, list or bool, default False
            True: 显示以下所有信息
            "iter_info": 一次优化过程的迭代信息
            "status": 优化过程收敛信息
            to be developed...
    ratio_max_holdings: 实际持股数偏离目标持股数的最大倍数, float, default 1.5
    constraints_relax: 约束放松选项, dict, default config_PM_BT中的相关设定
            该参数包含两个键值：step和limit，即某个约束条件以多大的步长（step）放松
        到多大的数值（limit），若优化仍不收敛则放弃继续优化
            目前约束只包含换手（to）和暴露（exposure）两种类型（先放松换手，若达到
        limit仍不收敛，则放松一次暴露，然后继续放松换手，依此类推），所有暴露的约束
        同时放松
    convergency_relax: 优化收敛放松选项, dict, default config_PM_BT中的相关设定
            当solver由于超过默认的最大迭代次数、迭代收敛容忍度等参数设定，而导致
        cvxpy抛出SolverError时，迭代次数在默认值基础上放大一定倍数"multiple_max_
        iters"，容忍度改为"tol2stop"
            注意，收敛放松的优先级高于约束放松，即当放松收敛条件后仍然优化失败时，才
        继续放松约束条件
    raise_SolverError: 强制抛出优化器异常, bool, default False
            该参数设置为真时，优化将直接按convergency_relax中的设定进行，节省按默认
        参数执行一遍的时间
    Methods:
    --------
    solve: 求解一期优化结果
        Parameters:
        -----------
        date: 交易日期
        w_lastday: 上期（交易前）持仓的相对权重, pandas.Series, default None
                index=股票代码
        stocks_suspended: 当期（交易时）停牌的股票, pandas.Series, default None
                回测时假设下一交易日的停牌个股都是能够提前获知的，因而该参数值为空时
                下一交易日的停牌个股的权重将保持上期的值
        Returns:
        --------
        dataframe(只含权重不为零的个股) or None(优化失败)
    """

    def __init__(
            self,
            signal,
            start_date=None,
            end_date=None,
            num_holdings=100,
            industry_neutral=None,
            industry_expourse=None,
            style_neutral=False,
            style_expourse=None,
            Epsilon=0.01,
            st_or_not=False,
            upper_bound=None,
            lower_bound=None,
            to_bound=None,
            te_bound=None,
            vol_bound=None,
            constituents_only=False,
            signal_direction=1,
            disp_info=False,
            ratio_max_holdings=1.5,
            constraints_relax=cfg.PARAMS_PM["constraints_relax"],
            convergency_relax=cfg.PARAMS_PM["convergency_relax"],
            raise_SolverError=False,
    ):
        # 判断信号方向
        msg_sd = "signal_direction should be 1 or -1"
        assert signal_direction in [1, -1], msg_sd
        self._signal_direction = signal_direction
        self.first = True
        self.industry_expourse = industry_expourse
        self.style_expourse = style_expourse
        self.st_or_not = st_or_not
        # 下载交易日历
        self._check_trday_data(start_date, end_date)
        # 数据准备
        self._prepare_data(
            start_date,
            end_date,
            style_neutral,
            industry_neutral
        )
        # 将信号的index转换成columns
        self._signal_to_ExpRet(signal)
        # 设置中性化约束阈值
        self._Eps = Epsilon
        #
        self._check_neutral_constraints(industry_neutral, style_neutral)
        # 设置股票数目
        self.num_holdings = num_holdings
        # 设置持股下限
        if lower_bound is None:
            self.lb = 0
        else:
            self.lb = lower_bound
        # 设置持股上限
        if upper_bound is None:
            if num_holdings is not None:  # XXX 适用线性目标函数
                self.ub = float(int(1e4 / num_holdings) / 1e4)
        else:
            self.ub = upper_bound

        self.to = to_bound  # 设置换手率约束
        self.te = te_bound  # 设置跟踪误差约束
        self.vol = vol_bound  # 设置组合波动约束
        self.constituents_only = constituents_only  # 设置成分股约束

        if not disp_info:
            self.disp_info = []
        elif disp_info == True:
            self.disp_info = ["iter_info", "status"]  # 'tracking_error'
        else:
            self.disp_info = flat(disp_info)

        self._mx_hld = ratio_max_holdings  # 实际持股数偏离目标持股数的最大倍数
        self._const_relax = constraints_relax  # 约束放松选项
        self._convg_relax = convergency_relax  # 优化收敛放松选项
        self._raise_se = raise_SolverError  # 强制抛出优化器异常

    def solve(self, date, w_lastday=None, stocks_suspended=None):
        return self._optimize_1day(date, w_lastday, stocks_suspended)

    def _check_neutral_constraints(self, industry_neutral, style_neutral):
        """检查和设置中性化约束"""
        msg = "%s supposed to be of type str, list or bool"
        neutralize = False

        self.style_neutral = cfg.style_exposure.copy()
        if isinstance(style_neutral, (str, list)):
            self.style_neutral.update({s: self._Eps for s in flat(style_neutral)})
            if not (self.style_expourse is None):
                self.style_neutral.update(
                    {s: self.style_expourse[s] for s in flat(self.style_expourse)}
                )
            neutralize = True
        elif isinstance(style_neutral, (bool, type(None))):
            if style_neutral:
                self.style_neutral.update(
                    {s: self._Eps for s in self.style_neutral.keys()}
                )
                neutralize = True
        else:
            raise KeyError(msg % "style_neutral")

        self.industry_neutral = cfg.industry_exposure.copy()
        if isinstance(industry_neutral, (str, list)):
            self.industry_neutral.update({i: self._Eps for i in flat(industry_neutral)})
            if not (self.style_expourse is None):
                self.industry_neutral.update(
                    {s: self.industry_expourse[s] for s in flat(self.industry_expourse)}
                )
            neutralize = True
        elif isinstance(industry_neutral, (bool, type(None))):
            if industry_neutral:
                self.industry_neutral.update(
                    {i: self._Eps for i in self.industry_neutral.keys()}
                )
                neutralize = True
        else:
            raise KeyError(msg % "industry_neutral")

        self._neutralize = neutralize

    #    @timer
    def _prepare_data(self, start_date, end_date, style_neutral, industry_neutral):
        """
        加载数据
        """
        if start_date is None:
            start_date = "2008-01-01"
        if end_date is None:
            end_date = "2099-12-31"

        # 行业信息
        tmp = fetch_data.fetch(start_date, end_date, [cfg.col_industries_name])[cfg.col_industries_name]
        tmp.fillna(0, inplace=True)
        tmp = tmp.astype(int)
        self.df_industries = tmp.copy()

        # 读取成分股权重
        self.df_weights = fetch_data.fetch(start_date, end_date, [cfg.col_index_component_weight])[cfg.col_index_component_weight] / 100.0

        self.d_factors = fetch_factor(start_date, end_date, standard='barra_alla')

        self.d_eodprices = fetch_data.fetch(start_date, end_date, cfg.cols_eodprices)

        # 中性化
        if style_neutral == True:  # 风格中性化
            self.cols_f = cfg.style_exposure.keys()
        elif style_neutral == False:
            self.cols_f = []
        else:

            self.cols_f = style_neutral

        if industry_neutral == True:
            self.cols_i = list(cfg.industry_exposure.keys())  # SW1级行业
        elif industry_neutral == False:
            self.cols_i = []
        else:
            self.cols_i = industry_neutral

    def _check_style_factors(self, date):
        """
        整理风格因子
        :param date:
        :return:
        """
        tmplst = []
        columns = []
        for col, tmpdf in self.d_factors.items():
            tmplst.append(tmpdf.loc[date, :])
            columns.append(col)
        newdf = pd.concat(tmplst, axis=1)
        newdf.columns = columns
        newdf.index.name = 'code'
        newdf.reset_index(inplace=True)
        return newdf

    def _check_industry_matrix(self, se_industries):
        """
        检验行业信息亚变量
        :param se_industries:
        :return:
        """
        tmp = pd.get_dummies(se_industries, prefix='D', columns=['industry_code'])
        try:
            tmp.drop("D_0", axis=1, inplace=True)
        except KeyError:
            pass
        return tmp.copy()

    def _check_style_matrix(self, df_factors):
        """
        检验风格亚变量，按照风格分组。分组可以考虑等分(pd.qcut)或者根据经验设置分组阈值 pd.cut()
        本函数实现等分qcut方法
        :param df_factors:  风格变量
        :return:
        """
        newlst = []
        for style_i in self.cols_f:
            bins = int(100 / cfg.style_categorical[style_i])
            tmplabels = [f'q_{i}' for i in range(0, 100, bins)]
            add_cols = f'{style_i}_group'
            df_factors[add_cols] = pd.qcut(df_factors[style_i], bins, labels=tmplabels)
            tmp = pd.get_dummies(df_factors[add_cols], prefix=style_i, columns=[add_cols])
            tmp.index = df_factors['code']
            newlst.append(tmp)
            self.style_neutral.update({s: self._Eps for s in tmp.columns})
        if len(newlst)==0:
            return df_factors.copy()
        elif len(newlst)==1:
            newdf = newlst[0]
            newdf.index.name = 'code'
            newdf.reset_index(inplace=True)
            return newdf.copy()
        else:
            newdf = pd.concat(newlst, axis=1)
            newdf.index.name = 'code'
            newdf.reset_index(inplace=True)
            return newdf.copy()

    def _check_eodprices_matrix(self, date):
        """
        将行情数据进行拼接
        :param date:
        :return:
        """
        tmplst = []
        columns = []
        for col, tmpdf in self.d_eodprices.items():
            tmplst.append(tmpdf.loc[date, :])
            columns.append(col)
        newdf = pd.concat(tmplst, axis=1)
        newdf.columns = columns
        newdf.index.name = 'code'
        newdf.reset_index(inplace=True)
        return newdf

    def _check_trday_data(self, start_date, end_date):
        """交易日时间序列"""
        trdays = query_data.get_trade_days('d', exchange='SSE', from_trade_day=start_date, to_trade_day=end_date)
        trdays.sort()
        self._tradedays = pd.to_datetime(trdays)

    def _signal_to_ExpRet(self, signal):
        """
        信号转换为预期收益
        (to be developed，目前由于优化目标函数简单设为线性函数， 无需对信号或打分
        做变换)
        """
        signal *= -self._signal_direction  # 将信号转为得分越低预期收益越高
        self.ExpRet = signal.copy()

    def _se2df(self, se, name, index_name='code'):
        """
        将series 转换成dataframe
        :param se: pd.Series
        :param name: str
        :param index_name: str default 'code'
        :return:
        """
        se.name = name
        se.index.name = index_name
        return se.reset_index()

    def _optimize_1day(self, date, w_lastday, stocks_suspended):
        """
        优化一天的持仓
        Parameters:
        -----------
        date: 交易日期, str
        w_lastday: 上个交易日持仓权重, pandas.Series
        stocks_suspended: 当前交易日停牌个股, iterable
        """

        date = pd.to_datetime(date)
        self._dt = date
        # 读取某一天的数据
        se_industries = self._se2df(self.df_industries.loc[date, :], 'industry_code')  # 行业代码
        df_industries = self._check_industry_matrix(se_industries)  # 行业亚变量
        df_index_weight = self._se2df(self.df_weights.loc[date, :], 'i_weight')  # 成分股权重
        df_ExpRet = self._se2df(self.ExpRet.loc[date, :], 'score')  # 个股打分,删除空值（可能是未来才上市，历史数据选取时间较长）
        df_ExpRet.dropna(how='any', axis=0, inplace=True)
        df_factors = self._check_style_factors(date)
        df_eodprices = self._check_eodprices_matrix(date)

        df_ExpRet = df_ExpRet[df_ExpRet['code'].isin(df_factors['code'])]
        if not self.st_or_not:
            no_st_stocklst = df_eodprices.loc[df_eodprices['st_or_not']!=1, 'code'].tolist()
            df_ExpRet = df_ExpRet[df_ExpRet['code'].isin(no_st_stocklst)]  # 过滤掉ST股票

        # 取预期收益（信号）、因子暴露及成分股的并集中的股票代码作为各数据对齐标准
        newdf_factors = self._check_style_matrix(df_factors)
        rawdata = pd.merge(df_industries, newdf_factors)
        rawdata.sort_values('code', inplace=True)
        if len(self.cols_f) > 0:
            f_cols = list(newdf_factors.columns)
            f_cols.remove('code')
            cols = self.cols_i + f_cols
            self.cols_stype = f_cols
        else:
            cols = self.cols_i
            self.cols_stype = []
        for col in cols:
            rawdata[col] = rawdata[col].fillna(rawdata[col].median())
        self.loading_1d = rawdata.set_index('code').loc[:, cols]

        w_b = pd.merge(df_ExpRet, df_index_weight, how='left')
        w_b['weight'] = (w_b["i_weight"] / w_b["i_weight"].sum()).fillna(0)
        self.w_b_1d = w_b.loc[:, ['code', 'weight']].set_index('code')  # 基准指数成分股权重重新调整

        self.eps = pd.Series({**self.industry_neutral, **self.style_neutral}).loc[cols].copy()

        stklist = df_ExpRet.copy()
        self.stklist = stklist['code'].copy()

        w0 = w_lastday
        if w0 is None:  # 不传入昨日持仓时，昨日持仓按0计
            w0 = stklist.loc[:, ['code', 'score']].copy()
            w0.rename(columns={'score': 'weight'}, inplace=True)
            w0['weight'] = 0
            self.w0 = w0.set_index('code')
        else:
            if isinstance(w0, pd.Series):
                w0 = w0.reset_index()
                w0.columns = ["code", "weight"]
            w0 = w0.merge(stklist, how="right").sort_values("code")
            self.w0 = w0.loc[:, ["code", "weight"]].fillna(0).set_index('code')

        # stk_notrd: 不能交易的个股，通过将其预期收益设为极差来避免对其持仓
        if stocks_suspended is None:
            stocks_suspended = df_eodprices.loc[df_eodprices["suspend"] == 1, "code"]
        # stk_nochg: 不做交易的个股，即出现在昨日持仓且今日停牌的个股，通过限制持仓上下限来保持持仓不变
        tmp_nochg = w0.copy()
        stk_nochg = tmp_nochg.loc[tmp_nochg["code"].isin(stocks_suspended), "code"]
        self.stk_nochg = stk_nochg.copy()
        df_ExpRet.loc[df_ExpRet["code"].isin(stk_nochg), "score"] = 1000000  # XXX: 将下一交易日停牌的个股的预期收益设为空值
        self.ExpRet_1d = df_ExpRet.set_index('code')
        return self._optimize()

    def _optimize(self):
        """
        优化模块， 当前版本可以做线性约束
        """
        w = cp.Variable(len(self.stklist))  # 今日持仓权重（全市场股票）
        stklist = self.stklist.tolist()
        F = self.loading_1d.loc[stklist, :].values  # 个股的因子暴露
        w_b = self.w_b_1d.loc[stklist, 'weight'].values  # 成分股权重
        exp_relative = F.T * (w - w_b)  # 相对基准指数的暴露
        exp_absolute = F.T * w  # 绝对暴露
        w0 = self.w0.loc[stklist, 'weight'].values  # 昨日持仓权重

        epsilon = cp.Parameter(F.shape[1], nonneg=True, value=self.eps.values)  # 中性化约束的值

        # upper bound
        ub = cp.Parameter(w.shape, nonneg=True)
        value = np.zeros(w.shape)
        idx_sus = [stklist.index(v) for v in self.stk_nochg.values]
        if self.constituents_only:  # 只持成分股时，非成分股上限设为0
            value[np.argwhere(w_b > 0)] = self.ub
        else:
            value[:] = self.ub
        value[idx_sus] = w0[idx_sus]  # 不交易的个股持仓上限设为昨日权重
        ub.value = value

        # lower bound
        lb = cp.Parameter(w.shape, nonneg=True)
        value = np.zeros(w.shape)
        value[idx_sus] = w0[idx_sus]  # 不交易的个股持仓下限设为昨日权重
        lb.value = value

        const = [cp.sum(w) == 1, w >= lb, w <= ub]

        # 中性化约束
        if self._neutralize:
            const.extend([exp_relative <= epsilon, exp_relative >= -epsilon])

        # 换手率约束
        if self.to is not None and self.first == False:
            to = cp.Parameter(nonneg=True)
            to.value = 1 if (w0 == 0).all() else self.to
            const.append(cp.norm(w - w0, 1) / 2 <= to)

        dt = self._dt.strftime("%Y-%m-%d")
        vb = "iter_info" in self.disp_info

        obj = self.ExpRet_1d.values.T * w  # + cp.quad_form(exp_relative, sigma) * 1e-6
        prob = cp.Problem(cp.Minimize(obj), const)

        # 换手率/中性化约束逐步放松
        for solver in ["ECOS", "SCS"]:  #
            raise_se = self._raise_se
            step = 0
            while prob.status != "optimal":
                try:
                    if raise_se:
                        raise SolverError(solver)
                    prob.solve(solver=solver, verbose=vb)
                    if prob.status:
                        if prob.status == "optimal":
                            break
                except DCPError:
                    raise
                except SolverError as e:
                    step = step + 1
                    raise_se = True
                    if "ECOS" in e.args[0]:
                        try:
                            prob.solve(
                                solver=solver,
                                verbose=vb,
                                max_iters=100 * self._convg_relax["multiple_max_iters"],
                                abstol=self._convg_relax["tol2stop"],
                                reltol=self._convg_relax["tol2stop"] / 10.0,
                                feastol=self._convg_relax["tol2stop"],
                            )
                        except:
                            pass
                    elif "SCS" in e.args[0]:
                        try:
                            prob.solve(
                                solver=solver,
                                verbose=vb,
                                max_iters=2500 * self._convg_relax["multiple_max_iters"],
                                eps=self._convg_relax["tol2stop"],
                            )
                        except:
                            pass

                if self.to is None or self.first == True:
                    break
                else:
                    to.value += self._const_relax["step"]["to"]
                    #                    print('to', to.value)
                    if to.value > self._const_relax["limit"]["to"]:
                        to.value = self.to

                        if self._neutralize:
                            epsilon.value += self._const_relax["step"]["exposure"]
                            #                            print('expo', epsilon.value.min())
                            if (epsilon.value.min()> self._const_relax["limit"]["exposure"]):
                                epsilon.value = self.eps
                                break

                        value = np.zeros(w.shape)
                        if self.constituents_only:  # 只持成分股时，非成分股上限设为0
                            if (
                                    self.ub
                                    + step * self._const_relax["step"]["upper_bound"]
                                    > self._const_relax["limit"]["upper_bound"]
                            ):
                                value[np.argwhere(w_b > 0)] = (
                                        self.ub
                                        + step * self._const_relax["step"]["upper_bound"]
                                )
                            else:
                                value[np.argwhere(w_b > 0)] = self.ub
                        else:
                            value[:] = self.ub
                        value[self.stk_nochg] = w0[self.stk_nochg]  # 不交易的个股持仓上限设为昨日权重
                        ub.value = value

                        if not self._neutralize and self.te is None:
                            break

            if prob.status == "optimal":
                break

        if "status" in self.disp_info:
            print("{}\t{}".format(dt, prob.status))

        if prob.status == "optimal":
            w_star = pd.DataFrame([stklist, w.value], index=["code", "weight"]).T
            w_star.loc[w_star["weight"] < 1e-4, "weight"] = 0

            num_last_holdings = np.sum(w0 > 0)

            if num_last_holdings > self.num_holdings * self._mx_hld:  # 优化结果的持股数量大于阈值时，强行调整持股数
                w_star = w_star.merge(self.ExpRet_1d.reset_index(), how="left")
                w_star.sort_values(["weight", "score", "code"],
                    ascending=[False, True, True],
                    inplace=True,
                )
                w_star.drop("score", axis=1, inplace=True)
                w_star["weight"].iloc[self.num_holdings:] = 0

            w_star["weight"] /= w_star["weight"].sum()
            w_star.set_index("code", inplace=True)

            expo = pd.DataFrame(
                F.T @ (w_star.values - w_b.reshape(len(w_b), 1)),
                index=self.cols_i + self.cols_stype,
                columns=[dt],
            ).T

            # XXX: only for debug purpose. inaccessible to user
            if "debug" in self.disp_info:
                print(
                    "{}\t{}\t{:.2f}\t{:.2f}\t{}".format(
                        dt,
                        prob.status,
                        # expo.at[dt, 'Size'],
                        # expo.at[dt, 'nl_size'],
                        expo.at[dt, "D_801050"],
                        abs(w_star.sort_index(ascending=True).values.ravel() - w0).sum()
                        / 2,
                        w.value.max(),
                        #                       te.value,
                        solver,
                    )
                )

            w_star = w_star["weight"]
            w_star = w_star[w_star.gt(1e-8) & w_star.notnull()]
        else:
            w_star = None
            expo = None
            print("optimization on %s failed" % dt)
        self.first = False
        return w_star, expo


if __name__ == '__main__':
    # test_portfolios()
    pass
