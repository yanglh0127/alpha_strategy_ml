import pandas as pd
import datetime as dt
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

pat = 'E:\\FT_Users\\LihaiYang\\svix'
data = pd.read_csv(pat+'\\sp500option_1996-2020.csv',engine='python')
data['now_date'] = pd.to_datetime(data['date'])
data['expire_date'] = pd.to_datetime(data['exdate'])

data['price'] = (data['best_bid']+data['best_offer'])/2
data['K'] = data['strike_price']/1000
data = data[['now_date','expire_date','cp_flag','K','price','best_bid','best_offer','volume','exercise_style']]
data['T'] = data['expire_date'] - data['now_date']
c = [int(i.days) for i in data['T']]
data['Tt'] = c
data = data.drop(['T'],1)

'''filter'''
option_data = data[data['exercise_style']=='E']
option_data = option_data.drop_duplicates(subset=['now_date','expire_date','K','cp_flag'],keep='first',inplace=False)
option_data = option_data[option_data['volume']>0]  # 此条件保留与不保留均测
option_data = option_data[(option_data['Tt']>=7)&(option_data['Tt']<=550)]
option_data = option_data[option_data['best_bid']>0]
option_data = option_data.dropna(axis=0,how='any')
option_data = option_data.reset_index(drop=True)
option_data = option_data.sort_values(by = ['now_date','expire_date','K','cp_flag'],axis = 0,ascending = True)
option_data = option_data.reset_index(drop=True)

'''计算隐含期货价格'''
option_data['op_dif'] = option_data['price'].groupby([option_data['now_date'],option_data['expire_date'],option_data['K']]).diff()
option_data['abs_opdif'] = abs(option_data['op_dif'])
a = option_data['abs_opdif'].groupby([option_data['now_date'],option_data['expire_date']]).min()
a = a.dropna(axis=0,how='any')
a = a.reset_index()
a = a.reset_index(drop=True)
a.rename(columns = {'abs_opdif':'min_abs_opdif'},inplace = True)
option_data1 = option_data.merge(a,on=['now_date','expire_date'],how='left')
option_data1 = option_data1[option_data1['abs_opdif'] == option_data1['min_abs_opdif']]
option_data1 = option_data1.drop_duplicates(subset=['now_date','expire_date'],keep='first',inplace=False)
rf_data = pd.read_csv(pat+"\\rf.csv",engine='python')
rf_data['now_date'] = pd.to_datetime(rf_data['date'])
rf_data = rf_data[['now_date','rf']]
option_data1 = option_data1.merge(rf_data,on='now_date',how='left')
option_data1 = option_data1.dropna(axis=0,how='any')
option_data1 = option_data1.reset_index(drop=True)
option_data1['forward'] = option_data1['K'] - option_data1['op_dif']*(1+option_data1['rf']*option_data1['Tt'])
option_data1 = option_data1[['now_date','expire_date','forward','rf']]
option_data = option_data.drop(['op_dif','abs_opdif'],1)
option_data = option_data.merge(option_data1,on=['now_date','expire_date'],how='left')
option_data = option_data.dropna(axis=0,how='any')
option_data = option_data.reset_index(drop=True)

'''同一个行权价选择call或put'''
option_data['type'] = option_data['cp_flag'].apply(lambda g: 1 if g=='P' else -1)
option_data['dis'] = option_data['forward'] - option_data['K']
option_data['sig'] = option_data['dis']*option_data['type']
option_data = option_data[option_data['sig']>=0]
option_data = option_data.dropna(axis=0,how='any')
option_data = option_data.reset_index(drop=True)

sp_data = pd.read_csv(pat+"\\sp500_2020.csv",engine='python')
sp_data['now_date'] = pd.to_datetime(sp_data['caldt'])
sp_data = sp_data[['now_date','spindx']]
comb_data = option_data.merge(sp_data,on='now_date',how='left')
comb_data = comb_data.dropna(axis=0,how='any')
comb_data = comb_data.sort_values(by = ['now_date','expire_date','K','cp_flag'],axis = 0,ascending = True)
comb_data = comb_data.reset_index(drop=True)
co = comb_data['K'].groupby([comb_data['now_date'],comb_data['expire_date'],comb_data['cp_flag']]).count()
co = co.reset_index()
cou = co['cp_flag'].groupby([co['now_date'],co['expire_date']]).count()
cou = cou.reset_index()
comb_data = comb_data.drop(['type','sig','dis'],1)

'''求premium'''
comb_data['k_before'] = comb_data['K'].groupby([comb_data['now_date'],comb_data['expire_date']]).shift(1)
comb_data['k_after'] = comb_data['K'].groupby([comb_data['now_date'],comb_data['expire_date']]).shift(-1)
comb_data['delta_k'] = comb_data.apply(lambda g: g['k_after'] - g['K'] if np.isnan(g['k_before']) else (g['K'] - g['k_before'] if np.isnan(g['k_after']) else (g['k_after'] - g['k_before'])/2), axis=1)
comb_data['part'] = comb_data['price']*comb_data['delta_k']*2/(comb_data['spindx']**2)
premium = comb_data['part'].groupby([comb_data['now_date'],comb_data['expire_date']]).sum()
premium = premium.reset_index()
premium['T'] = premium['expire_date'] - premium['now_date']
d = [int(i.days) for i in premium['T']]
premium['Tt'] = d
def cal_interp_re30(x):
    f = interpolate.interp1d(x['Tt'],x['part'],kind='linear',fill_value="extrapolate")
    return float(f(30))
def cal_interp_re60(x):
    f = interpolate.interp1d(x['Tt'],x['part'],kind='linear',fill_value="extrapolate")
    return float(f(60))
def cal_interp_re90(x):
    f = interpolate.interp1d(x['Tt'],x['part'],kind='linear',fill_value="extrapolate")
    return float(f(90))
def cal_interp_re180(x):
    f = interpolate.interp1d(x['Tt'],x['part'],kind='linear',fill_value="extrapolate")
    return float(f(180))
def cal_interp_re360(x):
    f = interpolate.interp1d(x['Tt'],x['part'],kind='linear',fill_value="extrapolate")
    return float(f(360))

premium = premium.sort_values(by = ['now_date','expire_date'],axis = 0,ascending = True)
premium = premium.reset_index(drop=True)
premium_30 = premium.groupby(['now_date']).apply(cal_interp_re30)
premium_30 = premium_30.reset_index()
premium_30['premium_30'] = premium_30[0]*365/30
premium_30 = premium_30.drop([0],1)
premium_60 = premium.groupby(['now_date']).apply(cal_interp_re60)
premium_60 = premium_60.reset_index()
premium_60['premium_60'] = premium_60[0]*365/60
premium_60 = premium_60.drop([0],1)
premium_90 = premium.groupby(['now_date']).apply(cal_interp_re90)
premium_90 = premium_90.reset_index()
premium_90['premium_90'] = premium_90[0]*365/90
premium_90 = premium_90.drop([0],1)
premium_180 = premium.groupby(['now_date']).apply(cal_interp_re180)
premium_180 = premium_180.reset_index()
premium_180['premium_180'] = premium_180[0]*365/180
premium_180 = premium_180.drop([0],1)
premium_360 = premium.groupby(['now_date']).apply(cal_interp_re360)
premium_360 = premium_360.reset_index()
premium_360['premium_360'] = premium_360[0]*365/360
premium_360 = premium_360.drop([0],1)
premium_standard = premium_30.merge(premium_60,on='now_date',how='left')
premium_standard = premium_standard.merge(premium_90,on='now_date',how='left')
premium_standard = premium_standard.merge(premium_180,on='now_date',how='left')
premium_standard = premium_standard.merge(premium_360,on='now_date',how='left')
premium_standard = premium_standard.merge(sp_data,on='now_date',how='left')
premium_standard = premium_standard.dropna(axis=0,how='any')
premium_standard = premium_standard.reset_index(drop=True)
premium_standard.to_excel(pat+'\\premium.xlsx',encoding='gbk',index=False)
# 画图
for k in ['premium_30', 'premium_60', 'premium_90', 'premium_180', 'premium_360']:
    fig, ax1 = plt.subplots(figsize = (10,8))
    ax2 = ax1.twinx()
    ax1.plot(premium_standard['now_date'], premium_standard[k],'r-')
    ax2.plot(premium_standard['now_date'], premium_standard['spindx'],'g-')
    plt.show()
