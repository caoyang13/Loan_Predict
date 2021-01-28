import numpy as np

lgb_result = np.loadtxt('./data/result_lightgbm.csv')
cat_result = np.loadtxt('./data/result_catboost.csv')

result = (lgb_result + cat_result) / 2

np.savetxt('./data/result.csv', result, fmt='%f')