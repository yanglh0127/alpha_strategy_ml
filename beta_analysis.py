import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import json

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


def beta_significant(bta):
    mea = {}
    tva = {}
    le = np.size(bta, 0)
    # la = math.ceil(4*(le/100)**(2/9))
    la = 480
    for fac_name in bta.columns:
        mea[fac_name] = sm.OLS(bta[fac_name], [1 for i in range(le)]).fit(cov_type='HAC', cov_kwds={'maxlags': la}).params[0]
        tva[fac_name] = sm.OLS(bta[fac_name], [1 for i in range(le)]).fit(cov_type='HAC', cov_kwds={'maxlags': la}).tvalues[0]
    return mea, tva

mea_ols, tva_ols = beta_significant(beta_ols['pool_480'])
insig_ols = {k: v for k, v in tva_ols.items() if abs(v) < 1.69}
with open(data_pat + "/beta_persistence/lag480/mea_ols.json", "w") as f:
    json.dump({k[0]: k[1] for k in sorted(mea_ols.items(), key=lambda x: abs(x[1]), reverse=True)}, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/tva_ols.json", "w") as f:
    json.dump(tva_ols, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/insig_ols.json", "w") as f:
    json.dump(insig_ols, f, indent=4)
print(len(insig_ols))

mea_ridge, tva_ridge = beta_significant(beta_ridge['pool_480_0.2'])
insig_ridge = {k: v for k, v in tva_ridge.items() if abs(v) < 1.69}
with open(data_pat + "/beta_persistence/lag480/mea_ridge.json", "w") as f:
    json.dump({k[0]: k[1] for k in sorted(mea_ridge.items(), key=lambda x: abs(x[1]), reverse=True)}, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/tva_ridge.json", "w") as f:
    json.dump(tva_ridge, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/insig_ridge.json", "w") as f:
    json.dump(insig_ridge, f, indent=4)
print(len(insig_ridge))

mea_lasso, tva_lasso = beta_significant(beta_lasso['pool_480_4e-05'])
insig_lasso = {k: v for k, v in tva_lasso.items() if abs(v) < 1.69}
with open(data_pat + "/beta_persistence/lag480/mea_lasso.json", "w") as f:
    json.dump({k[0]: k[1] for k in sorted(mea_lasso.items(), key=lambda x: abs(x[1]), reverse=True)}, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/tva_lasso.json", "w") as f:
    json.dump(tva_lasso, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/insig_lasso.json", "w") as f:
    json.dump(insig_lasso, f, indent=4)
print(len(insig_lasso))

mea_ela, tva_ela = beta_significant(beta_ela['pool_480_0.1_0.0004'])
insig_ela = {k: v for k, v in tva_ela.items() if abs(v) < 1.69}
with open(data_pat + "/beta_persistence/lag480/mea_ela.json", "w") as f:
    json.dump({k[0]: k[1] for k in sorted(mea_ela.items(), key=lambda x: abs(x[1]), reverse=True)}, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/tva_ela.json", "w") as f:
    json.dump(tva_ela, f, indent=4)
with open(data_pat + "/beta_persistence/lag480/insig_ela.json", "w") as f:
    json.dump(insig_ela, f, indent=4)
print(len(insig_ela))
