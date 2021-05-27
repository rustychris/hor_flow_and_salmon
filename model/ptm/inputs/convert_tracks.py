import pandas as pd

from stompy import utils

utils.path("../../../field/hor_yaps")
import track_common

##

master_in="../../../field/hor_yaps/screen_final"

df=track_common.read_from_folder(master_in)

# Make this look like Ed's input:
df_ptm=pd.DataFrame()
df_ptm['track']=df['basename'].str.replace('.csv','')
df_ptm['tag']=df.index.values

##

# Find the x,y for the time of entry:
df_ptm['entry_time']=np.nan # if 1st detection was above array
df_ptm['first_detection_time']=np.nan  # But not before array
df_ptm['exit_time']=np.nan
df_ptm['x']=np.nan
df_ptm['y']=np.nan
df_ptm['route']=""

for idx,rec in df.iterrows():
    entry_t=rec['top_of_array_first']
    if np.isnan(entry_t):
        x=rec['track']['x'][0]
        y=rec['track']['y'][0]
        first_detection_time=rec['track']['tnum'][0]
    else:
        x=np.interp(entry_t,rec['track']['tnum'].values,rec['track']['x'].values)
        y=np.interp(entry_t,rec['track']['tnum'].values,rec['track']['y'].values)
        first_detection_time=entry_t

    df_ptm.loc[idx,'x']=x
    df_ptm.loc[idx,'y']=y
    df_ptm.loc[idx,'entry_time']=entry_t
    df_ptm.loc[idx,'first_detection_time']=first_detection_time

    sj=np.isfinite(rec['sj_upper_first'])
    hor=np.isfinite(rec['hor_upper_first'])
    if sj and not hor:
        route='San_Joaquin'
        exit_t=rec['sj_upper_first']
    elif hor and not sj:
        route='Head_of_Old_River'
        exit_t=rec['hor_upper_first']
    elif hor and sj:
        if rec['sj_upper_first']<rec['hor_upper_first']:
            route='San_Joaquin'
            exit_t=rec['sj_upper_first']
        else:
            route='Head_of_Old_River'
            exit_t=rec['hor_upper_first']
    else:
        route='no_exit'
        exit_t=np.nan
    df_ptm.loc[idx,'route']=route
    df_ptm.loc[idx,'exit_time']=exit_t

##

# 20201230: Rename the epoch timestamps, and make the
# entry_time and exit_time as string datetimes in PST.

df_ptm2=df_ptm.copy()
utc_to_pst=np.timedelta64(-8,'h')

for t_col in ['entry_time','exit_time','first_detection_time']:
    epo_col=t_col.replace('_time','_utc_epoch')
    pst_col=t_col+"_pst"
    df_ptm2[epo_col]=df_ptm2[t_col]
    df_ptm2[pst_col] = utc_to_pst + utils.unix_to_dt64(df_ptm2[epo_col].round())
    del df_ptm2[t_col]

##

df_ptm2.to_csv("screen_final-ptm_inputs-20201230.csv",index=False)

