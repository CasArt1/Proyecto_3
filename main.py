import pandas as pd
from data_loader import download_data, load_data, plot_normalized, plot_prices
from cointegration_test import test_cointegration_johansen
from signal_generator import generate_trading_signals
from backtesting import backtest_strategy
from plot_signal import plot_trading_signals

if __name__ == "__main__":
    tickers = ["CDNS", "SNPS"]

   
    download_data(tickers)
    data = load_data().dropna()

    
    stock1, stock2 = data[tickers[0]], data[tickers[1]]
    plot_prices(data)
    plot_normalized(data,tickers)

    
    eigenvector = test_cointegration_johansen(stock1, stock2)

    hedge_ratio_johansen = eigenvector[1] / eigenvector[0] if eigenvector[0] != 0 else eigenvector[1]

    
    signals = generate_trading_signals(stock1, stock2, initial_hedge_ratio=hedge_ratio_johansen)


    plot_trading_signals(stock1, stock2, signals)
    backtest_strategy(signals)

    print("Backtesting completado.")