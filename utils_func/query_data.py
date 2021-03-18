# -*- coding: utf-8 -*-
# python3
"""
数据库读取接口函数
"""

import pandas as pd
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import pymongo
import requests, json
import re
import os
import types

# 基本函数
def get_productid(symbol):
    return re.match('[a-zA-Z]+', symbol)[0]

engines = {}

def init_engine(
        url=None, name="main",
        dialect=None, username=None, password=None, server=None, dbname=None,
        **kwargs):
    if url is None:
        url = '{}://{}:{}@{}/{}'.format(dialect, username, password, server, dbname)
    engine = create_engine(url, **kwargs)
    session_maker = sessionmaker(expire_on_commit=False)
    session_maker.configure(bind=engine)

    @contextmanager
    def _session_scope(maker):
        session = maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    engine.session_scope = types.MethodType(_session_scope, session_maker)
    engines[name] = engine
    return engine

def get_engine(name="main"):
    if name in engines:
        return engines[name]
    else:
        return None

class MongoDB:
    def __init__(self, host="192.168.1.99", port=27017, dbname='ftresearch', username='ftresearch', password='FTResearch', **kw):
        self.client = pymongo.MongoClient(f"mongodb://{host}:{port}/{dbname}", username=username, password=password, **kw)
        if dbname:
            self.db = self.client[dbname]
        else:
            self.db = self.client.test

    def close(self):
        return self.client.close()

    def __getattr__(self, key):
        return self.db[key]

    def __getitem__(self, key):
        return self.db[key]

    def has_collection(self, name):
        return name in self.db.list_collection_names()


class PostgreSQL():
    def __init__(self):
        self.mongodb = MongoDB()

    @staticmethod
    def get_engine(dbname):
        engine = get_engine(name=dbname)
        if engine is None:
            engine = init_engine(
                name=dbname,
                dialect="postgresql+psycopg2",
                server="192.168.1.99",
                username="ftresearch",
                password="FTResearch",
                dbname=dbname,
                echo=False)
        return engine

    def _get_tablename_1d(self, code):
        security = self.mongodb.securities.find_one({"code":code}, {"_id":0, "measurement":1} )
        if security and "measurement" in security:
            return security["measurement"]
        else:
            return None

    def _get_type(self, code):
        security = self.mongodb.securities.find_one({"code":code}, {"_id":0, "type":1})
        if security and "type" in security:
            return security["type"]
        else:
            return None

    def get_fields_1d(self, code, fromdate, todate, fields=None):
        tablename = self._get_tablename_1d(code)
        if not tablename:
            return None
        condition = f" where code='{code}'"
        if fromdate:
            condition += ' and'
            condition += f" trade_date>='{fromdate}'"
        if todate:
            condition += ' and'
            condition += f" trade_date<='{todate}'"
        if fields:
            columns = "trade_date," + ','.join(fields)
        else:
            columns = '*'
        sql = f'SELECT {columns} FROM {tablename} {condition}'
        # print(sql)
        df = pd.read_sql_query(sql, con=self.get_engine("ftresearch"))
        df.set_index("trade_date", inplace=True)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def _get_field_1d(self, field, code, fromdate, todate):
        tablename = self._get_tablename_1d(code)
        if not tablename:
            return None
        condition = f" where code='{code}'"
        if fromdate:
            condition += ' and'
            condition += f" trade_date>='{fromdate}'"
        if todate:
            condition += ' and'
            condition += f" trade_date<='{todate}'"
        sql = f'SELECT trade_date, {field} FROM {tablename} {condition}'
        df = pd.read_sql_query(sql, con=self.get_engine("ftresearch"))
        df.set_index("trade_date", inplace=True)
        if df is not None:
            s = df[field]
            s.name = code
            return s
        else:
            return None

    def get_field_1d(self, field, code_list, fromdate, todate):
        series_list = []
        for code in code_list:
            s = self._get_field_1d(field, code, fromdate, todate)
            if s is not None:
                series_list.append(s)
        df = pd.concat(series_list, axis=1)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def get_fields_1m(self, code, fromdate, todate, fields=None, night_period=True):
        code_type = self._get_type(code)
        if code_type == "future":
            return self.get_futures_fields_1m(code, fromdate, todate, fields, night_period)
        if code_type == "stock":
            return self.get_stock_fields_1m(code, fromdate, todate, fields)
        if code_type == "index":
            return self.get_index_fields_1m(code, fromdate, todate, fields)
        if code_type == "fund":
            return self.get_fund_fields_1m(code, fromdate, todate, fields)
        else:
            assert(False, "尚未实现")

    def get_field_1m(self, field, code_list, fromdate, todate, night_period=True):
        series_list = []
        for code in code_list:
            df = self.get_fields_1m(code, fromdate, todate, [field], night_period=night_period)
            if df is not None:
                s = df[field]
                s.name = code
                series_list.append(s)
        df = pd.concat(series_list, axis=1)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def get_futures_fields_1m(self, code, fromdate, todate, fields, night_period):
        tablename = f"futures_{get_productid(code).lower()}"
        condition = f" where code='{code}'"
        if fromdate:
            condition += ' and'
            if len(fromdate) <= 10:
                condition += f" time>='{prev_tradingday(fromdate, exchange='SHFE')} 21:00:00'"
            else:
                condition += f" time>='{fromdate}'"
        if todate:
            condition += ' and'
            if len(todate) <= 10:
                condition += f" time<'{todate} 16:00:00'"
            else:
                condition += f" time<='{todate}'"
        if fields:
            columns = "time," + ','.join(fields)
        else:
            columns = '*'
        sql = f'SELECT {columns} FROM {tablename} {condition}'
        # print(sql)
        df = pd.read_sql_query(sql, con=self.get_engine("futures1m"))
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        if not night_period:
            df = df.loc[(df.index.hour > 8) & (df.index.hour < 16), :]
        return df

    @staticmethod
    def _get_months(fromdate, todate):
        m = re.match(r"(\d{4})-(\d{2})", fromdate)
        if not m:
            m = re.match(r"(\d{4})(\d{2})", fromdate)
        from_year = int(m.group(1))
        from_month = int(m.group(2))

        m = re.match(r"(\d{4})-(\d{2})", todate)
        if not m:
            m = re.match(r"(\d{4})(\d{2})", todate)
        to_year = int(m.group(1))
        to_month = int(m.group(2))

        months = []
        year = max(from_year, 2005) #  数据从2005年开始
        month = from_month
        while year < to_year or (year == to_year and month <= to_month):
            months.append(f"{year:04d}{month:02d}")
            month += 1
            if month > 12:
                month -= 12
                year += 1
        current_month = datetime.datetime.now().strftime('%Y%m')
        months = [v for v in months if v <= current_month and v >= '200502']
        return months

    def get_stock_fields_1m(self, code, fromdate, todate, fields):
        df_all = None
        months = self._get_months(fromdate, todate)
        for month in months:
            tablename = f"stock_{month}"
            condition = f" where code='{code}'"
            if fromdate:
                condition += ' and'
                condition += f" time>='{fromdate}'"
            if todate:
                condition += ' and'
                if len(todate) <= 10:
                    condition += f" time<'{todate}'::timestamp+ interval'+1 day'"
                else:
                    condition += f" time<='{todate}'"
            if fields:
                columns = "time," + ','.join(fields)
            else:
                columns = '*'
            sql = f'SELECT {columns} FROM {tablename} {condition}'
            # print(sql)
            df = pd.read_sql_query(sql, con=self.get_engine("stock1m"))
            # print(df)
            if df_all is not None:
                df_all = pd.concat([df_all, df])
            else:
                df_all = df
        df_all.set_index("time", inplace=True)
        df_all.sort_index(inplace=True)
        df_all.index = pd.to_datetime(df_all.index)
        return df_all

    def get_stock_intraday_bar(self, field, fromdate, todate, freq):
        df_all = None
        months = self._get_months(fromdate, todate)
        for month in months:
            tablename = f"stock_intraday_bar_{month}"
            condition = f" where freq='{freq}'"
            if fromdate:
                condition += " and"
                condition += f" time>='{fromdate}'"
            if todate:
                condition += ' and'
                if len(todate) <= 10:
                    condition += f" time<'{todate}'::timestamp+ interval'+1 day'"
                else:
                    condition += f" time<='{todate}'"

            columns = f"time, code, {field}"
            sql = f'SELECT {columns} FROM {tablename} {condition}'
            # print(sql)
            df = pd.read_sql_query(sql, con=self.get_engine("stock_intraday"))
            # print(df)
            if df_all is not None:
                df_all = pd.concat([df_all, df])
            else:
                df_all = df
        newdf = pd.pivot_table(df_all, values=field, index=['time'], columns=['code'])
        newdf.sort_index(inplace=True)
        newdf.index = pd.to_datetime(newdf.index)
        newcols = [v[:6] for v in newdf.columns]
        newdf.columns = newcols
        return newdf.copy()

    def get_index_fields_1m(self, code, fromdate, todate, fields):
        df_all = None
        months = self._get_months(fromdate, todate)
        for month in months:
            tablename = f"index_{month}"
            condition = f" where code='{code}'"
            if fromdate:
                condition += ' and'
                condition += f" time>='{fromdate}'"
            if todate:
                condition += ' and'
                if len(todate) <= 10:
                    condition += f" time<'{todate}'::timestamp+ interval'+1 day'"
                else:
                    condition += f" time<='{todate}'"
            if fields:
                columns = "time," + ','.join(fields)
            else:
                columns = '*'
            sql = f'SELECT {columns} FROM {tablename} {condition}'
            # print(sql)
            df = pd.read_sql_query(sql, con=self.get_engine("index1m"))
            # print(df)
            if df_all is not None:
                df_all = pd.concat([df_all, df])
            else:
                df_all = df
        df_all.set_index("time", inplace=True)
        df_all.sort_index(inplace=True)
        df_all.index = pd.to_datetime(df_all.index)
        return df_all

    def get_fund_fields_1m(self, code, fromdate, todate, fields):
        df_all = None
        months = self._get_months(fromdate, todate)
        for month in months:
            tablename = f"fund_{month}"
            condition = f" where code='{code}'"
            if fromdate:
                condition += ' and'
                condition += f" time>='{fromdate}'"
            if todate:
                condition += ' and'
                if len(todate) <= 10:
                    condition += f" time<'{todate}'::timestamp+ interval'+1 day'"
                else:
                    condition += f" time<='{todate}'"
            if fields:
                columns = "time," + ','.join(fields)
            else:
                columns = '*'
            sql = f'SELECT {columns} FROM {tablename} {condition}'
            # print(sql)
            df = pd.read_sql_query(sql, con=self.get_engine("fund1m"))
            # print(df)
            if df_all is not None:
                df_all = pd.concat([df_all, df])
            else:
                df_all = df
        df_all.set_index("time", inplace=True)
        df_all.sort_index(inplace=True)
        df_all.index = pd.to_datetime(df_all.index)
        return df_all

    # 获取后复权因子，按照交易时间频率返回
    def _get_adjfactor(self, fromdate, todate):
        """通过日频的复权因子序列，填充成指定频率de复权因子序列"""
        sql = f"SELECT trade_date, code, adjfactor from ashareeodprices where trade_date>='{fromdate}' and trade_date<='{todate}'"
        rawdf = pd.read_sql_query(sql, con=self.get_engine("ftresearch"))
        df = pd.pivot_table(rawdf, values='adjfactor', index=['trade_date'], columns=['code'])
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        newindex = pd.date_range(start=fromdate, end=todate, freq='1T')
        f_index = [v for v in newindex if (v.hour==0 and v.minute==0) or (v.hour>=9 and v.hour<=15)]
        # newdf = pd.DataFrame(index=f_index, columns=df.columns)
        newdf = df.reindex(f_index, method='ffill')
        return newdf.copy()




# 查询接口
db_mongo = MongoDB()


def to_Y_m_d(d):
    if isinstance(d, datetime.datetime):
        return d.strftime('%Y-%m-%d')
    if isinstance(d, datetime.date):
        return d.strftime('%Y-%m-%d')
    if type(d) == pd.Timestamp:
        return d.strftime('%Y-%m-%d')
    if type(d) != str:
        d = f"{d}"
    if len(d) == 8:
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return d


def get_trade_days(freq, exchange="SSE", from_trade_day="1000-01-01", to_trade_day="2999-12-31", reverse=False):
    """
    获取交易时间序列
    :param freq: str [d', 'w', 'm', 'q'] 可选 日 周 月 季度
    :param exchange: str ['SSE', 'SHFE'] 上交所， 上期所
    :param from_trade_day: str ['yyyy-mm-dd' | 'yyyymmdd'] 开始日期（含）
    :param to_trade_day: str ['yyyy-mm-dd' | 'yyyymmdd'] 结束日期（含）
    :param reverse: bool [True, False] 正序，倒序
    :return: list format ['yyyy-mm-dd']
    """
    from_trade_day = to_Y_m_d(from_trade_day)
    to_trade_day = to_Y_m_d(to_trade_day)
    return [doc["trade_days"] for doc in
            db_mongo.calendar.find(
                {
                    "exchange": exchange,
                    "freq": freq,
                    "trade_days": {"$gte": from_trade_day, "$lte": to_trade_day}
                },
                {
                    "_id": 0,
                    "trade_days": 1
                }
            ).sort([("trade_days", (reverse and -1 or 1) )]) ]


def is_tradingday(tradingday, exchange="SSE", freq='d'):
    tradingday = to_Y_m_d(tradingday)
    r = db_mongo.calendar.find(
            {"exchange": exchange, "freq": freq, "trade_days": tradingday},
            {"_id": 0, "trade_days": 1}
        ).limit(1)
    for e in r:
        return True
    return False


def next_tradingday(tradingday, exchange="SSE", freq='d', lag=1):
    tradingday = to_Y_m_d(tradingday)
    r = db_mongo.calendar.find(
            {"exchange": exchange, "freq": freq, "trade_days": {"$gt": tradingday}},
            {"_id": 0, "trade_days": 1}
        ).sort([("trade_days", 1)]
        ).limit(lag)
    return [v["trade_days"] for v in r][-1]


def prev_tradingday(tradingday, exchange="SSE", freq='d', lag=1):
    tradingday = to_Y_m_d(tradingday)
    r = db_mongo.calendar.find(
            {"exchange": exchange, "freq": freq, "trade_days": {"$lt": tradingday}},
            {"_id": 0, "trade_days": 1}
        ).sort([("trade_days", -1)]
        ).limit(lag)
    return [v["trade_days"] for v in r][-1]


"""
Examples:
# 取出所有的=中金所的期货合约
get_security_info(type="future", exchange="cffex")
# 取出所有的科创板股票
get_security_info(type="stock", board_type='STARMarket')
# 取出 600000.SH
get_security_info(code="600000.SH")
"""
def get_security_info(**kw):
    """
    获取不同资产的合约信息
    :param type: str ['stock', 'future', 'option', 'index', 'ETF', 'fund'] 可选
    :param code: str 合约或股票代码 可选
    :return:
    """
    return [doc for doc in db_mongo.securities.find(kw,{"_id": 0})]

def get_alphafactors_info(**kw):
    """
    获取alpha因子的信息
    :param code: str 合约或股票代码 可选
    :return:
    """
    return [doc for doc in db_mongo.alpha_factors.find(kw,{"_id": 0})]

def rolling_data(se, field, windows):
    if field in ['close', 'oi']:
        return se.iloc[windows-1::windows]
    elif field in ['open', 'trade_status']:
        return se.shift(windows-1).iloc[windows-1::windows]
    elif field == 'low':
        return se.rolling(window=windows).min().iloc[windows-1::windows]
    elif field == 'high':
        return se.rolling(window=windows).max().iloc[windows-1::windows]
    elif field in ['volume', 'amt']:
        return se.rolling(window=windows).sum().iloc[windows-1::windows]

def make_bar(df, freq='5m', bar_t0=False, mult_fields=False, field=None):
    """
    由1分钟的数据合成其它频率的数据
    :param df: 1分钟数据
    :param freq: 其它频率
    :param bar_t0 Bool 是否限制在单个交易日内，数据不跨交易日
    :param mult_fields  Bool 是否包含多个字段
    :param field str 单只标获取时，指定字段名

    :return:
    """
    windows = int(freq[:-1])
    if bar_t0:
        t_days = list(set(df.index.strftime('%Y-%m-%d')))
        t_days.sort()
        newdf = pd.DataFrame()
        for t_day in t_days:
            start_time = f"{t_day} 08:00:00"
            end_time = f"{t_day} 16:00:00"
            tmpdf = df.loc[start_time:end_time, :]
            tmp = pd.DataFrame(index=tmpdf.index[windows-1::windows], columns=tmpdf.columns)
            if mult_fields:
                for col_f in tmpdf.columns:
                    tmp.update(rolling_data(tmpdf.loc[:, col_f], col_f, windows))
            else:
                tmp.update(rolling_data(tmpdf, field, windows))
            newdf = newdf.append(tmp)
    else:
        newdf = pd.DataFrame(index=df.index[windows-1::windows], columns=df.columns)
        if mult_fields:
            for col_f in df.columns:
                newdf.update(rolling_data(df.loc[:, col_f], col_f, windows))
        else:
            newdf.update(rolling_data(df, field, windows))
    return newdf

## Tick 查询

def query_tick_futures(tday='2020-08-25', night_period=True):
    """
    读取历史tick数据，从CTP下载的历史数据，按照交易日获取
    :param tday: str 'yyyy-mm-dd'
    :param night_period: Bool  是否包含夜盘区间，默认包含

    :return: pd.DataFrame() index 为数据落地时的本地时间  ClosePrice 1.7976931348623157e+308 为默认无效数据
    tradingDay 和ActionDay在不同交易所之间有差异，使用时注意
    """
    path_dir = r'\\192.168.1.111\mnt\srcData\CtpData'
    datestr = tday.replace('-', '')
    filelst = os.listdir(os.path.join(path_dir, datestr))
    f_file = [v for v in filelst if '.DepthMarketData.' in v]
    f_file.sort()
    columns = ['info', 'TradingDay', 'InstrumentID', 'ExchangeID', 'ExchangeInstID', 'LastPrice', 'PreSettlementPrice',
               'PreClosePrice', 'PreOpenInterest', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'Volume', 'Turnover',
               'OpenInterest', 'ClosePrice', 'SettlementPrice', 'UpperLimitPrice', 'LowerLimitPrice', 'PreDelta',
               'CurrDelta', 'UpdateTime', 'UpdateMillisec', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1',
               'BidPrice2', 'BidVolume2', 'AskPrice2', 'AskVolume2', 'BidPrice3', 'BidVolume3', 'AskPrice3',
               'AskVolume3', 'BidPrice4', 'BidVolume4', 'AskPrice4', 'AskVolume4', 'BidPrice5', 'BidVolume5',
               'AskPrice5', 'AskVolume5', 'AveragePrice', 'ActionDay']
    if night_period:
        tmplst = []
        for filename in f_file:
            tmppath = f'{path_dir}\{datestr}\{filename}'
            tmpdf = pd.read_csv(tmppath, index_col=1, names=columns)
            tmplst.append(tmpdf)
        if len(tmplst) > 1:
            newdf = pd.concat(tmplst, axis=0)
        else:
            newdf = tmpdf
    else:
        filename = f_file[-1]
        tmppath = f'{path_dir}\{datestr}\{filename}'
        newdf = pd.read_csv(tmppath, index_col=1, names=columns)
    return newdf


## 交易所网站爬虫
def get_mmf_hisunityield_sina(code, start_date='', end_date=''):
    """
    货币基金历史收益 sina
    :param code: '000389'
    :param start_date: 'yyyy-mm-dd'
    :return:
    """
    # url = r'http://api.fund.eastmoney.com/f10/lsjz?callback=mmf_unityield&fundCode=%s&pageIndex=1&pageSize=20&startDate=&endDate=&_=1600254056672'%code
    url = r'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNavcur?callback=mmf_unityield&symbol=%s&datefrom=%s&dateto=%s&page=1&_=1600254748904'% (code, start_date, end_date)
    data_r = requests.get(url)
    idx_start = data_r.text.index('(')
    data_j = json.loads(data_r.text[idx_start+1:-2])
    pages = int(data_j['result']['data']['total_num']) // 20+1
    newlst = []
    newdf = pd.DataFrame()
    for page in range(pages):
        url = r'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNavcur?callback=mmf_unityield&symbol=%s&datefrom=%s&dateto=%s&page=%s&_=1600254748904' % (
        code, start_date, end_date, page+1)
        data_r = requests.get(url)
        idx_start = data_r.text.index('(')
        data_j = json.loads(data_r.text[idx_start + 1:-2])
        tmpdf = pd.DataFrame(data_j['result']['data']['data'])
        if 'fbrq' not in tmpdf.columns:
            tmpdf = tmpdf.T
        newlst.append(tmpdf)
        # newdf = newdf.append(tmpdf)
    columns = {'fbrq': 'trade_date', 'NAV_CUR1': 'nav_cur1', 'nhsyl': 'mmf_annualizedyield', 'dwsy': 'mmf_unityield'}
    if len(newlst) == 1:
        newdf = pd.DataFrame(newlst[0])
    elif len(newlst) == 0:
        return None
    else:
        newdf = pd.concat(newlst, axis=0)
    if len(newdf) > 0:
        newdf.rename(columns=columns, inplace=True)
        newdf['mmf_annualizedyield'] = newdf['mmf_annualizedyield'].astype('float')
        newdf['mmf_unityield'] = newdf['mmf_unityield'].astype('float')
        newdf['mmf_annualizedyield'] = newdf['mmf_annualizedyield']/100.0
        newdf['code'] = f'{code}.OF'
        newdf.set_index('trade_date', inplace=True)
        newdf.index = pd.to_datetime(newdf.index)
        return newdf

def get_fund_hisnav_sina(code, start_date='', end_date=''):
    """
        公募基金历史净值 sina
        :param code: '510500'
        :param start_date: 'yyyy-mm-dd'
        :return:
        """
    # url = r'http://api.fund.eastmoney.com/f10/lsjz?callback=mmf_unityield&fundCode=%s&pageIndex=1&pageSize=20&startDate=&endDate=&_=1600254056672'%code
    url = r'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNav?callback=hisnav&symbol=%s&datefrom=%s&dateto=%s&page=1&_=1600312047214' % (
    code, start_date, end_date)
    data_r = requests.get(url)
    idx_start = data_r.text.index('(')
    data_j = json.loads(data_r.text[idx_start + 1:-2])
    pages = int(data_j['result']['data']['total_num']) // 20 + 1
    newlst = []
    newdf = pd.DataFrame()
    for page in range(pages):
        url = r'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNav?callback=hisnav&symbol=%s&datefrom=%s&dateto=%s&page=%s&_=1600312047214' % (
            code, start_date, end_date, page + 1)
        data_r = requests.get(url)
        idx_start = data_r.text.index('(')
        data_j = json.loads(data_r.text[idx_start + 1:-2])
        tmpdf = pd.DataFrame(data_j['result']['data']['data'])
        if 'fbrq' not in tmpdf.columns:
            tmpdf = tmpdf.T
        newlst.append(tmpdf)
        # newdf = newdf.append(tmpdf)
    columns = {'fbrq': 'trade_date', 'jjjz': 'nav', 'ljjz': 'nav_acc'}
    if len(newlst) == 1:
        newdf = pd.DataFrame(newlst[0])
    elif len(newlst) == 0:
        return None
    else:
        newdf = pd.concat(newlst, axis=0)
    if len(newdf) > 0:
        newdf.rename(columns=columns, inplace=True)
        newdf['nav'] = newdf['nav'].astype('float')
        newdf['nav_acc'] = newdf['nav_acc'].astype('float')
        newdf['code'] = f'{code}.OF'
        newdf.sort_values('trade_date', inplace=True)
        newdf.set_index('trade_date', inplace=True)
        newdf.index = pd.to_datetime(newdf.index)
        return newdf


## 数据接口部分
def get_fields_fund(code, fromdate='2020-01-01', todate='2020-12-31', data_type='nav'):
    """
    获取公募基金单位净值或者货币基金收益率
    :param code: '510050'  不带后缀
    :param fromdate: 'yyyy-mm-dd' 获取全部历史时保持空 ''
    :param todate: 'yyyy-mm-dd' 获取全部历史时保持空 ''
    :param data_type: yield 货币基金收益率 nav 基金净值
    :return: df
    """
    if '.' in code:
        code = code.split('.')[0]
    if data_type == 'nav':
        return get_fund_hisnav_sina(code, start_date=fromdate, end_date=todate)
    elif data_type == 'yield':
        return get_mmf_hisunityield_sina(code, start_date=fromdate, end_date=todate)
    else:
        return print('请选择数据类型')

def get_fields(code, fromdate='20200101', todate='22000101', fields=None, freq='1d', bar_t0=False, night_period=True):
    """
    获取单合约的时间序列数据
    :param code: str 合约代码 '000001.SZ'
    :param fromdate:  str 开始日期 '20200101' 或者 '2020-01-01'
    :param todate: str 截至日期 '20200101' 或者 '2020-01-01'
    :param fields: list fields为None时，返回所有价格数据， 例如fields=["high", "open", "low", "close"]
    :param freq: str '1d'日频 '1m' 分钟频
    :param bar_t0 Bool 默认为False 是否限制在单个交易日内，False 表示不限制 True表示按照每个交易日合成
    :param night_period Boll 是否包含夜盘 True 包含夜盘时间 False 不包含夜盘时间 分钟数据有效
    :return:
    """
    db_postgre = PostgreSQL()
    if freq == '1d':
        return db_postgre.get_fields_1d(code, fromdate=fromdate, todate=todate, fields=fields)
    elif freq == '1m':
        return db_postgre.get_fields_1m(code, fromdate=fromdate, todate=todate, fields=fields, night_period=night_period)
    elif 'm' in freq and (int(freq[:-1]) < 240):
        df = db_postgre.get_fields_1m(code, fromdate=fromdate, todate=todate, fields=fields, night_period=night_period)
        return make_bar(df, freq=freq, bar_t0=bar_t0, mult_fields=True)

def get_field(field, fromdate='20200101', todate='22000101', code_list=['600000.SH'], freq='1d', bar_t0=False, night_period=True):
    """
    获取截面数据，多个合约同一字段
    :param field: str 字段名 'close', 'pe'
    :param fromdate:  str 开始日期 '20200101' 或者 '2020-01-01'
    :param todate: str 截至日期 '20200101' 或者 '2020-01-01'
    :param code_list: list 非空 例如code_list=['600000.SH', '000001.SZ']
    :param freq: str '1d'日频 '1m' 分钟频 '5m' 通过1分钟合成 '5min' 通过tick数据合成
    :param bar_t0 Bool 默认为False 是否限制在单个交易日内，False 表示不限制 True表示按照每个交易日合成
    :param night_period Boll 是否包含夜盘 True 包含夜盘时间 False 不包含夜盘时间 分钟数据有效
    :return:
    """
    db_postgre = PostgreSQL()
    if field == 'adjfactor' and 'm' in freq:
        return db_postgre._get_adjfactor(fromdate=fromdate, todate=todate)  # 返回1分钟间隔的后复权因子
    if freq == '1d':
        return db_postgre.get_field_1d(field, fromdate=fromdate, todate=todate, code_list=code_list)
    elif freq == '1m':
        return db_postgre.get_field_1m(field, fromdate=fromdate, todate=todate, code_list=code_list, night_period=night_period)
    elif 'min' in freq:
        return db_postgre.get_stock_intraday_bar(field, fromdate=fromdate, todate=todate, freq=freq)
    elif ('m' == freq[-1]) and (int(freq[:-1]) < 240):
        df = db_postgre.get_field_1m(field, fromdate=fromdate, todate=todate, code_list=code_list, night_period=night_period)
        return make_bar(df, freq=freq, bar_t0=bar_t0, mult_fields=False, field=field)

def get_tick(tday, d_type='future', night_period=True):
    """
    获取Tick数据
    :param tday: str 'yyyy-mm-dd'
    :param d_type: str  资产类型 future 期货CTP share 股票
    :param night_period: 夜盘数据标识，默认读取夜盘
    :return: df
    """
    if d_type == 'future':
        return query_tick_futures(tday=tday, night_period=night_period)
    elif d_type=='stock':
        #\\192.168.1.111\mnt\srcData\XcData
        return '未实现'

if __name__ == '__main__':
    db = PostgreSQL()

    # r = db.get_fields_1d('600000.SH', fromdate="20190701", todate="20190710", fields=["high", "open", "low", "close"])
    # print(r)
    # r = db.get_fields_1d('000001.SZ', fromdate="20190701", todate="20190710")
    # print(r)
    # r = db.get_field_1d('close', code_list=['600000.SH', '000001.SZ'], fromdate="20190701", todate="20190710")
    # print(r)
    # r = db.get_fields_1m('PB1602.SHF', fromdate="20160101", todate="20161231", fields=["high", "open", "low", "close"])
    # print(r)
    # r = db.get_fields_1m('000001.SH', fromdate="2019-03-15 09:35:00", todate="2019-03-15 09:40:00", fields=["high", "open", "low", "close"])
    # print(r)
    # r = db.get_fields_1m('000001.SZ', fromdate="20190315", todate="20190614", fields=["high", "open", "low", "close"])
    # print(r)
    # df = get_fields('000001.SZ', fields=['close', 'open', 'high', 'low', 'volume', 'trade_status'], fromdate="2020-01-01", todate="2020-09-20",
    #                freq='5m')  # 返回两只股票的收盘价， 分钟频
    #
    # print(df.head())
    # df = get_fields_fund('510500.SH', fromdate='2020-01-01', todate='2020-08-31', data_type='nav')
    # print(df.head())
    #
    # df = get_field('open', fromdate="20160107", todate="20160107", freq='5min')
    # print(df)


    infos = get_alphafactors_info(factor_rating='A')
    print(infos[0])
    # df = db._get_adjfactor(fromdate='20201201', todate='20201222')
    # print(len(df))