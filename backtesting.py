import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(signals, initial_capital=1_000_000, com=0.00125, borrow_rate=0.0025, 
                   take_profit=0.15, stop_loss=0.05, max_position_pct=0.40, 
                   max_drawdown_pct=0.10, daily_loss_limit_pct=0.05):
    """
    Backtest pairs trading strategy with strict risk management:
    - Portfolio-level stop loss of 5%
    - Maximum position size of 40% per pair (80% total exposure)
    - Maximum drawdown limit of 10%
    - Daily loss limit of 5%
    """
    """
    Backtest a pairs trading strategy with proper risk management.
    
    Parameters:
    - max_position_pct: Maximum position size as percentage of capital (e.g., 0.20 = 20%)
    - max_drawdown_pct: Maximum allowed drawdown before stopping trading (e.g., 0.25 = 25%)
    - daily_loss_limit_pct: Maximum daily loss before stopping trading (e.g., 0.05 = 5%)
    """
    capital = initial_capital
    available_capital = initial_capital
    active_long = []
    active_short = []
    portfolio_values = [initial_capital] * len(signals)  # List to track portfolio value over time
    daily_borrow_rate = borrow_rate / 252
    
    # Risk management tracking
    peak_capital = initial_capital
    daily_pl = 0  # Track daily profit/loss
    last_trade_date = None
    portfolio_value = initial_capital  # Current portfolio value
    
    # Trade statistics tracking
    trades_history = []
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    total_trades = 0
    invested_capital = 0  # Track how much capital is invested

    for i in range(len(signals)):
        date = signals.index[i]
        price_AMD = signals["stock1_price"].iloc[i]  # AMD
        price_MSFT = signals["stock2_price"].iloc[i]  # MSFT
        z_score = signals["z_score"].iloc[i]

        # Calculate total portfolio P&L percentage before checking stops
        portfolio_pl_pct = 0
        total_position_value = 0
        
        # Calculate P&L for long positions
        for pos in active_long:
            current_value = price_AMD * pos["shares"]
            position_pl = current_value - (pos["shares"] * pos["bought_at"])
            total_position_value += current_value
            portfolio_pl_pct += position_pl / initial_capital
            
        # Calculate P&L for short positions
        for pos in active_short:
            current_value = price_MSFT * pos["shares"]
            position_pl = (pos["sold_at"] - price_MSFT) * pos["shares"]
            total_position_value += current_value
            portfolio_pl_pct += position_pl / initial_capital
            
        # Check stop loss on entire portfolio
        if portfolio_pl_pct <= -stop_loss:
            # Close all positions if portfolio stop loss is hit
            for pos in active_long[:]:
                profit = (price_AMD - pos["bought_at"]) * pos["shares"] * (1 - com)
                available_capital += pos["shares"] * price_AMD * (1 - com)
                
                trade_result = {
                    'type': 'long',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["bought_at"],
                    'exit_price': price_AMD,
                    'profit_pct': ((price_AMD - pos["bought_at"]) / pos["bought_at"]) * 100,
                    'profit_abs': profit,
                    'shares': pos["shares"],
                    'holding_period': (date - pos["start_date"]).days,
                    'exit_reason': 'portfolio_stop_loss'
                }
                trades_history.append(trade_result)
                
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    losing_trades += 1
                    total_loss += abs(profit)
            active_long.clear()
            
            # Close all short positions
            for pos in active_short[:]:
                profit = (pos["sold_at"] - price_MSFT) * pos["shares"] * (1 - com)
                available_capital += pos["shares"] * price_MSFT * (1 - com)
                
                trade_result = {
                    'type': 'short',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["sold_at"],
                    'exit_price': price_MSFT,
                    'profit_pct': ((pos["sold_at"] - price_MSFT) / pos["sold_at"]) * 100,
                    'profit_abs': profit,
                    'shares': pos["shares"],
                    'holding_period': (date - pos["start_date"]).days,
                    'exit_reason': 'portfolio_stop_loss'
                }
                trades_history.append(trade_result)
                
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    losing_trades += 1
                    total_loss += abs(profit)
            active_short.clear()
            continue
            
        # Individual position management
        for pos in active_long[:]:
            profit_pct = (price_AMD - pos["bought_at"]) / pos["bought_at"]
            if profit_pct >= take_profit or profit_pct <= -stop_loss:
                profit = (price_AMD - pos["bought_at"]) * pos["shares"] * (1 - com)
                available_capital += pos["shares"] * price_AMD * (1 - com)
                
                trade_result = {
                    'type': 'long',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["bought_at"],
                    'exit_price': price_AMD,
                    'profit_pct': profit_pct * 100,
                    'profit_abs': profit,
                    'shares': pos["shares"],
                    'holding_period': (date - pos["start_date"]).days,
                    'exit_reason': 'individual_stop'
                }
                trades_history.append(trade_result)
                
                # Update statistics
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    losing_trades += 1
                    total_loss += abs(profit)
                
                active_long.remove(pos)

        for pos in active_short[:]:
            profit_pct = (pos["sold_at"] - price_MSFT) / pos["sold_at"]
            if profit_pct >= take_profit or profit_pct <= -stop_loss:
                profit = (pos["sold_at"] - price_MSFT) * pos["shares"] * (1 - com)
                days_held = (date - pos["start_date"]).days
                total_borrow_cost = price_MSFT * pos["shares"] * daily_borrow_rate * days_held
                net_profit = profit - total_borrow_cost
                capital += net_profit + (pos["shares"] * pos["sold_at"])
                
                # Record trade statistics
                trade_result = {
                    'type': 'short',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["sold_at"],
                    'exit_price': price_MSFT,
                    'profit_pct': profit_pct,
                    'profit_abs': net_profit,
                    'shares': pos["shares"],
                    'holding_period': days_held,
                    'borrow_cost': total_borrow_cost
                }
                trades_history.append(trade_result)
                
                # Update statistics
                total_trades += 1
                if net_profit > 0:
                    winning_trades += 1
                    total_profit += net_profit
                else:
                    total_loss += abs(net_profit)
                
                active_short.remove(pos)

        # Reset daily P/L if it's a new day
        if last_trade_date is None or date.date() != last_trade_date.date():
            daily_pl = 0
        last_trade_date = date
        
        # Check risk management limits
        current_drawdown = (peak_capital - portfolio_values[i-1]) / peak_capital if i > 0 else 0
        if current_drawdown > max_drawdown_pct or daily_pl < -initial_capital * daily_loss_limit_pct:
            continue  # Skip trading if risk limits are breached
            
        # Señal de entrada
        if z_score > 1.5:
            # Always close any open positions before opening new ones
            for pos in active_long:
                profit = (price_AMD - pos["bought_at"]) * pos["shares"] * (1 - com)
                capital += profit + (pos["shares"] * pos["bought_at"])
            active_long.clear()
            for pos in active_short:
                profit = (pos["sold_at"] - price_MSFT) * pos["shares"] * (1 - com)
                capital += profit + (pos["shares"] * pos["sold_at"])
            active_short.clear()

            # Calculate position size using 40% of available capital per position
            portfolio_value = available_capital
            # Position size as 40% of portfolio value
            position_size = portfolio_value * max_position_pct

            # Calculate final share counts
            shares_AMD = int(position_size / price_AMD)
            shares_MSFT = int(position_size / price_MSFT)

            # Comprar AMD
            total_cost_AMD = price_AMD * shares_AMD * (1 + com)
            if available_capital >= total_cost_AMD:
                available_capital -= total_cost_AMD
                invested_capital += total_cost_AMD
                active_long.append({
                    "ticker": "AMD",
                    "bought_at": price_AMD,
                    "shares": shares_AMD,
                    "start_date": date,
                    "cost": total_cost_AMD
                })

            # Venta corta de MSFT
            total_cost_MSFT = price_MSFT * shares_MSFT * (1 + com)  # Include commission
            if available_capital >= total_cost_MSFT:
                available_capital -= total_cost_MSFT
                invested_capital += total_cost_MSFT
                active_short.append({
                    "ticker": "MSFT",
                    "sold_at": price_MSFT,
                    "shares": shares_MSFT,
                    "start_date": date,
                    "cost": total_cost_MSFT
                })

            # Calculate daily borrow cost for short positions
            for short_pos in active_short:
                borrow_cost = price_MSFT * short_pos["shares"] * daily_borrow_rate
                capital -= borrow_cost

        # Señal de cierre
        elif abs(z_score) < 0.5:
            # Cerrar largos (AMD)
            for pos in active_long:
                profit = (price_AMD - pos["bought_at"]) * pos["shares"] * (1 - com)
                capital += profit + (pos["shares"] * pos["bought_at"])
                
                # Record trade statistics
                trade_result = {
                    'type': 'long',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["bought_at"],
                    'exit_price': price_AMD,
                    'profit_pct': ((price_AMD - pos["bought_at"]) / pos["bought_at"]) * 100,
                    'profit_abs': profit,
                    'shares': pos["shares"],
                    'holding_period': (date - pos["start_date"]).days
                }
                trades_history.append(trade_result)
                
                # Update statistics
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    losing_trades += 1
                    total_loss += abs(profit)
            active_long.clear()

            # Cerrar cortos (MSFT)
            for pos in active_short:
                profit = (pos["sold_at"] - price_MSFT) * pos["shares"] * (1 - com)
                days_held = (date - pos["start_date"]).days
                total_borrow_cost = price_MSFT * pos["shares"] * daily_borrow_rate * days_held
                net_profit = profit - total_borrow_cost
                capital += net_profit + (pos["shares"] * pos["sold_at"])
                
                # Record trade statistics
                trade_result = {
                    'type': 'short',
                    'entry_date': pos["start_date"],
                    'exit_date': date,
                    'entry_price': pos["sold_at"],
                    'exit_price': price_MSFT,
                    'profit_pct': ((pos["sold_at"] - price_MSFT) / pos["sold_at"]) * 100,
                    'profit_abs': net_profit,
                    'shares': pos["shares"],
                    'holding_period': days_held,
                    'borrow_cost': total_borrow_cost
                }
                trades_history.append(trade_result)
                
                # Update statistics
                if net_profit > 0:
                    winning_trades += 1
                    total_profit += net_profit
                else:
                    losing_trades += 1
                    total_loss += abs(net_profit)
            active_short.clear()

        # Calculate current portfolio value including open positions
        current_portfolio = available_capital
        
        # Add value of long positions
        for pos in active_long:
            current_portfolio += pos["shares"] * price_AMD
            
        # Add value of short positions (minus borrowed shares)
        for pos in active_short:
            current_portfolio += pos["cost"] - (pos["shares"] * price_MSFT)
        
        portfolio_values[i] = current_portfolio
        portfolio_value = current_portfolio
        
        # Update peak capital for drawdown calculation
        if current_portfolio > peak_capital:
            peak_capital = current_portfolio
            
        # Update daily P/L
        if i > 0:
            daily_pl += portfolio_values[i] - portfolio_values[i-1]

    # Resultado final
    final_value = available_capital  # Start with available cash
    
    # Add value of any remaining positions
    for pos in active_long:
        final_value += pos["shares"] * signals["stock1_price"].iloc[-1]
    for pos in active_short:
        final_value += pos["cost"] - (pos["shares"] * signals["stock2_price"].iloc[-1])
    # Calculate final statistics
    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate max drawdown
    max_drawdown_pct = 0
    peak = initial_capital
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
    
    # Calculate average holding period
    avg_holding_period = sum(trade['holding_period'] for trade in trades_history) / len(trades_history) if trades_history else 0
    
    # Print statistics
    # Trade statistics already calculated above
    
    print("\n=== Trading Statistics ===")
    print(f"🎯 Total Trades: {total_trades}")
    print(f"✅ Winning Trades: {winning_trades}")
    print(f"❌ Losing Trades: {losing_trades}")
    print(f"📊 Win Rate: {win_rate:.2f}%")
    print(f"💰 Total Profit: ${total_profit:,.2f}")
    print(f"📉 Total Loss: ${total_loss:,.2f}")
    print(f"📈 Average Profit per Winning Trade: ${avg_profit:,.2f}")
    print(f"📉 Average Loss per Losing Trade: ${avg_loss:,.2f}")
    print(f"⚖️ Profit Factor: {(total_profit/total_loss if total_loss > 0 else 0):.2f}")
    print(f"📊 Max Drawdown: {max_drawdown_pct:.2f}%")
    print(f"⏱️ Average Holding Period: {sum(t['holding_period'] for t in trades_history)/total_trades:.1f} days")
    print(f"💼 Final Portfolio Value: ${final_value:,.2f}")
    print(f"📊 Total Return: {((final_value - initial_capital) / initial_capital * 100):.2f}%")

    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(signals.index, portfolio_values, label="Valor del Portafolio", color="blue")
    plt.title("Evolución del Portafolio")
    plt.xlabel("Fecha")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'portfolio_value': portfolio_value,
        'trades_history': trades_history,
        'statistics': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown_pct,
            'avg_holding_period': avg_holding_period,
            'final_value': final_value,
            'total_return': ((final_value - initial_capital) / initial_capital * 100)
        }
    }