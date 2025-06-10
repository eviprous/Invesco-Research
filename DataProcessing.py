import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
import statsmodels.api as sm
import wrds


class SP500DataHandler:
    def __init__(self, wrds_conn, start = '2005-01-01', end = '2025-05-31', frequency = 'M'):
        self.conn = wrds_conn
        self.start = start
        self.end = end
        self.frequency = frequency

        # Raw data pulls
        self.raw_returns = None
        self.raw_market_caps = None
        self.sp500_membership = None

        # Cleaned/filtered data
        self.cleaned_returns = None
        self.cleaned_market_caps = None

    def fetch_returns(self):
        '''Fetch SP_500 returns from WRDS'''
        self.raw_returns = self.conn.raw_sql(f"""
            SELECT permno, date, ret
            FROM crsp.msf 
            WHERE date BETWEEN '{self.start}' and '{self.end}'
        """)

    def fetch_sp500_membership(self):
        """Fetch historical S&P 500 membership data."""
        self.sp500_membership = self.conn.raw_sql("""
            SELECT permno, start, ending
            FROM crsp.dsp500list
        """)

    def fetch_market_caps(self):
        """Fetch prices and shares outstanding to compute market cap."""
        self.raw_market_caps = self.conn.raw_sql(f"""
            SELECT permno, date, prc, shrout
            FROM crsp.msf
            WHERE date BETWEEN '{self.start}' AND '{self.end}'
        """)
        self.raw_market_caps['me'] = self.raw_market_caps['prc'].abs() * self.raw_market_caps['shrout']

    def clean_and_merge(self):
        """Generate two separate cleaned datasets: returns and market cap."""
        # Filter returns to S&P 500 constituents
        if self.raw_returns is not None:
            ret_df = self.raw_returns.merge(self.sp500_membership, on='permno', how='left')
            ret_df = ret_df[(ret_df['date'] >= ret_df['start']) & (ret_df['date'] <= ret_df['ending'])]
            self.cleaned_returns = ret_df[['permno', 'date', 'ret']].copy()

        # Filter market cap to S&P 500 constituents
        if self.raw_market_caps is not None:
            mc_df = self.raw_market_caps.merge(self.sp500_membership, on='permno', how='left')
            mc_df = mc_df[(mc_df['date'] >= mc_df['start']) & (mc_df['date'] <= mc_df['ending'])]
            self.cleaned_market_caps = mc_df[['permno', 'date', 'me']].copy()

    def get_returns_pivot(self):
        """Pivot table of returns: rows = date, columns = permno."""
        return self.cleaned_returns.pivot(index='date', columns='permno', values='ret')

    def get_market_caps_pivot(self):
        """Pivot table of market caps: rows = date, columns = permno."""
        return self.cleaned_market_caps.pivot(index='date', columns='permno', values='me')

    def add_tickers(self, df):
        """Map PERMNOs in a pivoted DataFrame to tickers."""
        names = self.conn.raw_sql("""
            SELECT permno, namedt, nameendt, ticker
            FROM crsp.msenames
        """)
        names = names[names['nameendt'] >= self.start]
        ticker_map = names.sort_values('namedt').groupby('permno')['ticker'].last()
        df.columns = df.columns.map(lambda x: ticker_map.get(x, f"PERMNO_{x}"))
        return df

    def make_csv(self, filename="sp500_returns_panel.csv"):
        """Create CSV file from the cleaned returns panel."""
        if not hasattr(self, 'cleaned_returns'):
            raise ValueError("cleaned_returns is missing. Run clean_and_merge() first.")

        self.cleaned_returns.to_csv(filename, index=False)
        print(f"Saved returns panel to: {filename}")


if __name__ == '__main__':
    conn = wrds.Connection()

    handler = SP500DataHandler(conn)
    handler.fetch_sp500_membership()
    handler.fetch_market_caps()
    handler.clean_and_merge()

    cap_panel = handler.get_market_caps_pivot()
    cap_with_tickers = handler.add_tickers(cap_panel)

    cap_with_tickers.to_csv("sp500_market_caps.csv")
    print(cap_with_tickers.columns)


