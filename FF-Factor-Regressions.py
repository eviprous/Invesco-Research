import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt



tickers = {
    "SPC": "^GSPC",  # Cap-weighted index
    "SPE": "RSP"     # Equal-weight ETF
}
start_date = "2005-01-01"
end_date = "2025-05-31"

# Get adjusted monthly close
df = yf.download(list(tickers.values()), start=start_date, end=end_date, auto_adjust=True, interval="1mo")
close_df = df["Close"].copy()
close_df.columns = tickers.keys()
close_df.dropna(inplace=True)

returns_df = close_df.pct_change().dropna()
returns_df["Innovation Portfolio"] = returns_df["SPC"] - returns_df["SPE"]
returns_df = returns_df.iloc[:-1]

print(returns_df.tail())


#upload ff 3 factor data

ff_df = pd.read_csv("F-F_Research_Data_Factors-2.csv", index_col=0, skiprows=3)
ff_df = ff_df[ff_df.index.astype(str).str.match(r"^\d{6}$")]
ff_df.index = pd.to_datetime(ff_df.index.astype(str), format="%Y%m")
ff_df = ff_df.reindex(returns_df.index).dropna()

ff_df = ff_df.apply(pd.to_numeric, errors='coerce') / 100

print(ff_df.tail())
print(ff_df.dtypes)

merged_df = returns_df.join(ff_df, how='left')

print(merged_df.head(10))

excess_rets_df = pd.DataFrame()
excess_rets_df["SPC"] = merged_df["SPC"] - merged_df["RF"]
excess_rets_df["SPE"] = merged_df["SPE"] - merged_df["RF"]
excess_rets_df["Innovation Portfolio"] = merged_df["Innovation Portfolio"] - merged_df["RF"]
excess_rets_df["Mkt-RF"] = merged_df["Mkt-RF"]
excess_rets_df["SMB"] = merged_df["SMB"]
excess_rets_df["HML"] = merged_df["HML"]

print(excess_rets_df.head(10))

def compute_rolling_betas_and_alpha(
    excess_returns_df,
    portfolio_col,
    window=12,
    beta_MKT=False,
    beta_SMB=True,
    beta_HML=True,
    beta_RMW=False,
    beta_CMA=False,
    beta_Momentum=False
):
    """computes rolling betas and alphas"""
    factor_flags = {
        "Mkt-RF": beta_MKT,
        "SMB": beta_SMB,
        "HML": beta_HML,
        "RMW": beta_RMW,
        "CMA": beta_CMA,
        "Mom": beta_Momentum
    }

    selected_factors = [factor for factor, include in factor_flags.items() if include]

    results = {"alpha": []}
    for factor in selected_factors:
        results[f"beta_{factor}"] = []

    index = []

    for i in range(window, len(excess_returns_df)):
        window_df = excess_returns_df.iloc[i - window:i]

        y = window_df[portfolio_col]
        X = window_df[selected_factors]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        results["alpha"].append(model.params.get("const", float("nan")))
        for factor in selected_factors:
            results[f"beta_{factor}"].append(model.params.get(factor, float("nan")))

        index.append(excess_returns_df.index[i])

    return pd.DataFrame(results, index=index)

rolling_factors_innovation = compute_rolling_betas_and_alpha(excess_rets_df, "Innovation Portfolio", window=36)
rolling_factors_SPE = compute_rolling_betas_and_alpha(excess_rets_df, "SPE", window=36)
rolling_factors_SPC = compute_rolling_betas_and_alpha(excess_rets_df, "SPC", window=36)

#Print and show betas over time
print(rolling_factors_innovation.head())

rolling_factors_innovation.plot(title="12-Month Rolling Alpha and Betas: Innovation Portfolio")
plt.xlabel("Date")
plt.ylabel("Factor Exposure / Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()

rolling_factors_SPE.plot(title="12-Month Rolling Alpha and Betas: SPE")
plt.xlabel("Date")
plt.ylabel("Factor Exposure / Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()

rolling_factors_SPC.plot(title="12-Month Rolling Alpha and Betas: SPC")
plt.xlabel("Date")
plt.ylabel("Factor Exposure / Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()



### Rolling Mean, Standard Dev, and Sharpe ###

# Innovation excess returns
innovation_excess = excess_rets_df["Innovation Portfolio"]

# Rolling mean and std (monthly)
rolling_mean_monthly = innovation_excess.rolling(window=12).mean()
rolling_std_monthly = innovation_excess.rolling(window=12).std()

# Annualized mean and std
rolling_mean_annual = rolling_mean_monthly * 12
rolling_std_annual = rolling_std_monthly * np.sqrt(12)

# Sharpe ratio
rolling_sharpe_annual = rolling_mean_annual / rolling_std_annual

# Combine into DataFrame
rolling_stats_annual = pd.DataFrame({
    "Annualized Mean": rolling_mean_annual,
    "Annualized Std Dev": rolling_std_annual,
    "Annualized Sharpe": rolling_sharpe_annual
})

rolling_stats_annual.dropna(inplace=True)
print(rolling_stats_annual.head(10))

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

rolling_stats_annual["Annualized Mean"].plot(ax=axes[0], title="12-Month Rolling Annualized Mean Return")
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_ylabel("Mean")

rolling_stats_annual["Annualized Std Dev"].plot(ax=axes[1], title="12-Month Rolling Annualized Volatility")
axes[1].set_ylabel("Volatility")

rolling_stats_annual["Annualized Sharpe"].plot(ax=axes[2], title="12-Month Rolling Annualized Sharpe Ratio")
axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[2].set_ylabel("Sharpe")

plt.xlabel("Date")
plt.tight_layout()
plt.show()

print(rolling_stats_annual["Annualized Mean"].max())
print(rolling_stats_annual["Annualized Std Dev"].max())
print(rolling_stats_annual["Annualized Sharpe"].max())

print(rolling_stats_annual["Annualized Mean"].min())
print(rolling_stats_annual["Annualized Std Dev"].min())
print(rolling_stats_annual["Annualized Sharpe"].min())


# --------------------------
import pandas_datareader.data as web


# VIX from Yahoo
vix_df = yf.download("^VIX", start=start_date, end=end_date, interval="1mo")["Close"]
vix_df = vix_df.loc[returns_df.index.intersection(vix_df.index)]
vix_df_lagged = vix_df.shift(1) / 100  # Now units = decimal → matches rolling_vol

# 10Y and 2Y yields from FRED
dgs10 = web.DataReader('DGS10', 'fred', start_date, end_date)
dgs2 = web.DataReader('DGS2', 'fred', start_date, end_date)

# Combine into rates_df
rates_df = pd.concat([dgs10, dgs2], axis=1)
rates_df.columns = ["10Y", "2Y"]


# Convert index to monthly frequency to match your returns_df
rates_df = rates_df.resample('MS').last()
rates_df = rates_df / 100
print(rates_df.head())

# Compute Term Spread
rates_df["Term Spread"] = rates_df["10Y"] - rates_df["2Y"]


# Align to returns_df dates
rates_df = rates_df.loc[returns_df.index.intersection(rates_df.index)]

# Innovation Portfolio rolling volatility (decimal)
innovation = returns_df.loc[rates_df.index]["Innovation Portfolio"]
rolling_vol = innovation.rolling(window=12).std() * np.sqrt(12)  # Annualized volatility (decimal)


#plot the rolling vol
common_index = rolling_vol.index.intersection(vix_df_lagged.index)

plt.figure(figsize=(12, 6))
plt.plot(rolling_vol.loc[common_index], label="Innovation Portfolio Volatility (12M rolling, decimal)", color='blue')
plt.plot(vix_df_lagged.loc[common_index], label="VIX (Lagged 1 month, decimal)", color='red', linestyle='--')

plt.title("Innovation Portfolio Volatility vs. VIX (Both in decimal)")
plt.ylabel("Annualized Volatility (decimal)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#plots of vol and vix and term spread
common_index = rolling_vol.index.intersection(rates_df.index)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top plot: Innovation Portfolio Volatility
axes[0].plot(rolling_vol.loc[common_index], color='blue')
axes[0].set_ylabel('Innovation Volatility (annualized, decimal)')
axes[0].set_title('Innovation Portfolio Rolling Volatility (12M rolling)')

# Identify periods where Term Spread is negative
negative_spread_dates = rates_df.loc[common_index][rates_df["Term Spread"] < 0].index

# Plot vertical lines on volatility plot
for date in negative_spread_dates:
    axes[0].axvline(x=date, color='red', linestyle=':', alpha=0.5)

# Bottom plot: Term Spread
axes[1].plot(rates_df["Term Spread"].loc[common_index], color='green', linestyle='--')
axes[1].axhline(0, color='red', linestyle=':', linewidth=1.5)  # Horizontal line at 0
axes[1].set_ylabel('Term Spread (%)')
axes[1].set_title('Term Spread (10Y - 2Y)')

plt.xlabel("Date")
plt.tight_layout()
plt.show()


#correlation
# Make sure VIX is a Series
vix_series = vix_df_lagged.squeeze()

# Correlation of returns vs VIX
correlation_return_vix = returns_df["Innovation Portfolio"].loc[vix_series.index].corr(vix_series)
print(f"Correlation between Innovation Portfolio returns and VIX (Lagged): {correlation_return_vix:.3f}")

correlation = returns_df["Innovation Portfolio"].loc[rates_df.index].corr(rates_df["Term Spread"])
print(f"Correlation between Innovation Portfolio returns and Term Spread: {correlation:.3f}")



rolling_window = 12  # in months

# Common index to align everything
common_index = rolling_vol.index.intersection(vix_series.index).intersection(rates_df.index)

# Create aligned Series
vol_series = rolling_vol.loc[common_index]
vix_aligned = vix_series.loc[common_index]
termspread_aligned = rates_df.loc[common_index]["Term Spread"]

# Rolling correlations
rolling_corr_vol_vix = vol_series.rolling(window=rolling_window).corr(vix_aligned)
rolling_corr_vol_termspread = vol_series.rolling(window=rolling_window).corr(termspread_aligned)

print(rolling_corr_vol_vix)
print(rolling_corr_vol_termspread)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(rolling_corr_vol_vix, label='Rolling Corr: Innovation Vol vs VIX', color='red')
plt.plot(rolling_corr_vol_termspread, label='Rolling Corr: Innovation Vol vs Term Spread', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.title(f"Rolling {rolling_window}-Month Correlation of Innovation Volatility")
plt.ylabel("Correlation")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

