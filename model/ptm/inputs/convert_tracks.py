import pandas as pd

from stompy import utils

utils.path("../../../field/hor_yaps")
import track_common

##

master_in="../../../field/hor_yaps/screen_final"

df=track_common.read_from_folder(master_in)

df['track'].apply(track_common.calc_velocities,
                  model_u='model_u_top2m',
                  model_v='model_v_top2m')


##


# Make this look like Ed's input:
df_ptm=pd.DataFrame()
df_ptm['track']=df['basename'].str.replace('.csv','')
df_ptm['tag']=df.index.values
## 
# Find the x,y for the time of entry:
df_ptm['entry_time']=np.nan
df_ptm['exit_time']=np.nan
df_ptm['x']=np.nan
df_ptm['y']=np.nan
df_ptm['route']=""

for idx,rec in df.iterrows():
    entry_t=rec['top_of_array_first']
    if np.isnan(entry_t):
        x=y=np.nan
    else:
        x=np.interp(entry_t,rec['track']['tnum'].values,rec['track']['x'].values)
        y=np.interp(entry_t,rec['track']['tnum'].values,rec['track']['y'].values)
    df_ptm.loc[idx,'x']=x
    df_ptm.loc[idx,'y']=y

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
    
df_ptm.to_csv("screen_final-ptm_inputs-20201228.csv",index=False)
