### Cointegration Test

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint, adfuller

def adf_test(series):
    
    p_value = adfuller(series.dropna())[1]
    return p_value  

def test_cointegration_ols(stock1, stock2):

    p_value = coint(stock1, stock2)[1] 
    hedge_ratio = sm.OLS(stock1, sm.add_constant(stock2)).fit().params[1]
    return p_value, hedge_ratio

def test_cointegration_johansen(stock1, stock2):

    result = coint_johansen(pd.concat([stock1, stock2], axis=1).dropna(), det_order=0, k_ar_diff=1)
    return result.evec[:, 0] 