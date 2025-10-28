from data_loader import download_data, load_data
from cointegration_test import adf_test, test_cointegration_ols, test_cointegration_johansen

tickers = ["CDNS", "SNPS"]

    
download_data(tickers)
data = load_data().dropna()

stock1, stock2 = data[tickers[0]], data[tickers[1]]

print(adf_test(stock1))
print(adf_test(stock2))
print(test_cointegration_ols(stock1,stock2))
print(test_cointegration_johansen(stock1,stock2))