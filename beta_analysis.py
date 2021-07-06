import pandas as pd

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

beta_ols = pd.read_pickle(data_pat + '/ols/coef_pool.pkl')
des_ols = beta_ols['pool_480'].describe()

beta_ridge = pd.read_pickle(data_pat + '/ridge/coef_0.2.pkl')
des_ridge = beta_ridge['pool_480_0.2'].describe()

beta_lasso = pd.read_pickle(data_pat + '/lasso/coef_4e-05.pkl')
des_lasso = beta_lasso['pool_480_4e-05'].describe()

beta_ela = pd.read_pickle(data_pat + '/elnet/coef_0.1_0.0004.pkl')
des_ela = beta_ela['pool_480_0.1_0.0004'].describe()

def cs_data(bta):
    bta_now = bta.stack()
    bta_now.name = 'beta_now'
    bta_old = bta.shift(480).stack()
    bta_old.name = 'beta_old'
    bta_comb = pd.concat([bta_now, bta_old], axis=1)
    bta_comb = bta_comb.dropna(how='any')
    bta_comb = bta_comb.reset_index()
    bta_comb = bta_comb.sort_values(by=['level_1', 'level_0'])
    bta_comb = bta_comb.reset_index(drop=True)
    return bta_comb

btacomb_ols = cs_data(beta_ols['pool_480'])
btacomb_ols.to_csv(data_pat + '/beta_persistence/btacomb_ols.csv',encoding='gbk')

btacomb_ridge = cs_data(beta_ridge['pool_480_0.2'])
btacomb_ridge.to_csv(data_pat + '/beta_persistence/btacomb_ridge.csv',encoding='gbk')

btacomb_lasso = cs_data(beta_lasso['pool_480_4e-05'])
btacomb_lasso.to_csv(data_pat + '/beta_persistence/btacomb_lasso.csv',encoding='gbk')

btacomb_ela = cs_data(beta_ela['pool_480_0.1_0.0004'])
btacomb_ela.to_csv(data_pat + '/beta_persistence/btacomb_ela.csv',encoding='gbk')
