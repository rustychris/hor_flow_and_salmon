"""
Download and process flow data for Mossdale, to supply to survey.py

Not automation as much as just documenting what was done.
"""

import pandas as pd

url2018="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV"

df=pd.read_csv(url2018,skiprows=2,parse_dates=['Date'])

df['q_cms']=df['Point'] * 0.028316847
df['dnum']=utils.to_dnum(df.Date)

df.loc[:, ['dnum','Date','q_cms']].to_csv('msd_flow_2018.csv',index=False)
