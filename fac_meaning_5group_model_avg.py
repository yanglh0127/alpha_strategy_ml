from ft_platform.utils import utils_calculation as uc
import pandas as pd
import pickle


data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

fac_model = {}

ols_pred = pd.read_pickle(data_pat + '/ols/fac_pool.pkl')
fac_model['ols'] = uc.cs_rank(ols_pred['pool_480'])

ridge_pred = pd.read_pickle(data_pat + '/ridge/fac_0.2.pkl')
fac_model['ridge'] = uc.cs_rank(ridge_pred['pool_480_0.2'])

lasso_pred = pd.read_pickle(data_pat + '/lasso/fac_4e-05.pkl')
fac_model['lasso'] = uc.cs_rank(lasso_pred['pool_480_4e-05'])

elnet_pred = pd.read_pickle(data_pat + '/elnet/fac_0.1_0.0004.pkl')
fac_model['elnet'] = uc.cs_rank(elnet_pred['pool_480_0.1_0.0004'])

logit_pred = pd.read_pickle(data_pat + '/logit/fac_none.pkl')
fac_model['logit'] = uc.cs_rank(logit_pred['pool_480_none'])

nbayes_pred = pd.read_pickle(data_pat + '/bayes/fac.pkl')
fac_model['nbayes'] = uc.cs_rank(nbayes_pred['pool_480'])

pls_pred = pd.read_pickle(data_pat + '/pls/fac_6.pkl')
fac_model['pls'] = uc.cs_rank(pls_pred['pool_480_6'])

rf_pred = pd.read_pickle(data_pat + '/random_forest/fac_300_10_0.6.pkl')
fac_model['rf'] = uc.cs_rank(rf_pred['pool_480_300_10_0.6'])

gbdt_pred = pd.read_pickle(data_pat + '/gradient_boost/fac_300_5_0.1.pkl')
fac_model['gbdt'] = uc.cs_rank(gbdt_pred['pool_480_300_5_0.1'])

nn_pred = pd.read_pickle(data_pat + '/neural_network/fac_(6,)_256_0.001.pkl')
fac_model['nn'] = uc.cs_rank(nn_pred['pool_480_(6,)_256_0.001'])

# 模型平均
fac_all_eq = {}
comb = pd.concat(fac_model.values())
fac_all_eq['all_eq'] = comb.groupby(comb.index).mean()
fac_all_eq['all_eq'].index = pd.to_datetime(fac_all_eq['all_eq'].index)
f = open(data_pat + '/model_avg/fac_all_eq.pkl', 'wb')  # 记得修改
pickle.dump(fac_all_eq, f, -1)
f.close()

fac_nonlinear_eq = {}
fac_sub = {k: v for k, v in fac_model.items() if k in ['rf', 'gbdt', 'nn']}
comb1 = pd.concat(fac_sub.values())
fac_nonlinear_eq['nonlinear_eq'] = comb1.groupby(comb1.index).mean()
fac_nonlinear_eq['nonlinear_eq'].index = pd.to_datetime(fac_nonlinear_eq['nonlinear_eq'].index)
f = open(data_pat + '/model_avg/fac_nonlinear_eq.pkl', 'wb')  # 记得修改
pickle.dump(fac_nonlinear_eq, f, -1)
f.close()

fac_lnl_eq = {}
fac_sub1 = {k: v for k, v in fac_model.items() if k in ['pls', 'nn']}
comb2 = pd.concat(fac_sub1.values())
fac_lnl_eq['lnl_eq'] = comb2.groupby(comb2.index).mean()
fac_lnl_eq['lnl_eq'].index = pd.to_datetime(fac_lnl_eq['lnl_eq'].index)
f = open(data_pat + '/model_avg/fac_lnl_eq.pkl', 'wb')  # 记得修改
pickle.dump(fac_lnl_eq, f, -1)
f.close()
