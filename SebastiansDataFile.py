import pandas as pd
import wrds

# Establish connection to WRDS
conn = wrds.Connection()
# Insert credentials

# Fetch S&P 500 data
sp500 = conn.raw_sql("""
    SELECT a.*, b.date, b.ret
    FROM crsp.msp500list AS a
    JOIN crsp.msf AS b
    ON a.permno = b.permno
    WHERE b.date >= a.start 
    AND b.date <= a.ending
    AND b.date >= '01/01/1990'
    ORDER BY b.date;
    """, date_cols=['start', 'ending', 'date'])

# Add Other Company Identifiers from CRSP.MSENAMES
mse = conn.raw_sql("""
    SELECT comnam, ncusip, namedt, nameendt, 
           permno, shrcd, exchcd, hsiccd, ticker
    FROM crsp.msenames
    """, date_cols=['namedt', 'nameendt'])

# If nameendt is missing then set it to today's date
mse['nameendt'] = mse['nameendt'].fillna(pd.to_datetime('today'))

# Merge with S&P 500 data
sp500_full = pd.merge(sp500, mse, how='left', on='permno')

# Impose the date range restrictions
sp500_full = sp500_full.loc[
    (sp500_full.date >= sp500_full.namedt) &
    (sp500_full.date <= sp500_full.nameendt)
]

# Add Compustat Identifiers
ccm = conn.raw_sql("""
    SELECT gvkey, liid AS iid, lpermno AS permno,
           linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE SUBSTR(linktype, 1, 1) = 'L'
    AND (linkprim = 'C' OR linkprim = 'P')
    """, date_cols=['linkdt', 'linkenddt'])

# If linkenddt is missing then set it to today's date
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# Merge the CCM data with S&P 500 data
sp500_ccm = pd.merge(sp500_full, ccm, how='left', on='permno')

# Impose the link date bounds
sp500_ccm = sp500_ccm.loc[
    (sp500_ccm['date'] >= sp500_ccm['linkdt']) &
    (sp500_ccm['date'] <= sp500_ccm['linkenddt'])
]

# Drop unnecessary columns
sp500_ccm = sp500_ccm.drop(columns=['namedt', 'nameendt', 'linktype', 'linkprim', 'linkdt', 'linkenddt'])

# Rearrange columns for final output
final_columns = [
    'date', 'permno', 'comnam', 'ncusip', 'shrcd', 'exchcd',
    'hsiccd', 'ticker', 'gvkey', 'iid', 'start', 'ending', 'ret'
]
sp500_ccm = sp500_ccm[final_columns]

# Ensure PERMNO is an integer
sp500_ccm['permno'] = sp500_ccm['permno'].astype(int)

# Save to CSV
sp500_ccm.to_csv('sp500.csv', index=False)