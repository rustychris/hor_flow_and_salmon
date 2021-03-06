"""
Timeline figure
  Tag release
  Tag arrival - cumulative count
  Flows during study period
  Turbidity(?) during study period
  ADCP transecting period
"""
from stompy import utils
from stompy.spatial import field, wkb2shp
import stompy.plot.cmap as scmap
from stompy.plot import plot_wkb, plot_utils
from stompy.grid import unstructured_grid
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr 

import os, glob
from stompy import memoize
from stompy.io.local import cdec
from scipy import signal 

yaps_dir="../../../field/hor_yaps"
utils.path(yaps_dir)

import track_common

utils.path("../../../model/suntans")
import common
##

df_start=track_common.read_from_folder(os.path.join(yaps_dir,'screen_final'))
t_min=df_start.track.apply( lambda t: t.tnum.min() )
t_max=df_start.track.apply( lambda t: t.tnum.min() )
t_mid=0.5*(t_min+t_max)
df_start['t_mid']=t_mid

##
@memoize.memoize(lru=10)
def fetch_and_parse(local_file,url,**kwargs):
    if not os.path.exists(local_file):
        if not os.path.exists(os.path.dirname(local_file)):
            os.makedirs(os.path.dirname(local_file))
        utils.download_url(url,local_file,on_abort='remove')
    return pd.read_csv(local_file,**kwargs)

## 

tag_releases=np.array([np.datetime64("2018-03-02 00:00"),
                       np.datetime64("2018-03-15 00:00")] )
              
##

# Out of curiosity, how do arrivals compare between the
# releases?
tag_ids=df_start.index.values
tagged_fish=pd.read_excel('../../../field/circulation/2018_Data/2018FriantTaggingCombined.xlsx')
real_fish_tags=tagged_fish.TagID_Hex
upper=tagged_fish['Release Date'] < np.datetime64("2018-03-04")
lower=~upper # some NaT, mostly 2018-03-15
tagged_fish['release']=np.where(upper,"upper","lower")
## 
# Add release column to df_start, then
# split df_start into df_upper and df_lower
df_with_release=df_start.merge( tagged_fish[ ['TagID_Hex','release']], left_index=True,
                                right_on='TagID_Hex')

# 103 of the tracks were from the lower release.
# 30 tracks from the upper release.
print( df_with_release.groupby('release').size() )

df_upper=df_with_release[ df_with_release['release']=='upper']
df_lower=df_with_release[ df_with_release['release']=='lower']

##

times=utils.unix_to_dt64(df_start.t_mid.values)
pad=np.timedelta64(1,'D')
t_min=min(times.min(),tag_releases.min()) - pad
t_max=max(times.max(),tag_releases.max()) + pad

# MSD flows
msd_flow=common.msd_flow(np.datetime64("2018-03-01"),
                         np.datetime64("2018-04-20"))
# fetch_and_parse(local_file="env_data/msd/flow-2018.csv",
#                          url="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV",
#                          skiprows=3,parse_dates=['time'],names=['time','flow_cfs','quality','notes'])

msd_turb=cdec.cdec_dataset('MSD',t_min,t_max,sensor=27,cache_dir='env_data')
msd_turb['turb_lp']=('time',),signal.medfilt(msd_turb.sensor0027.values,5)

msd_temp=cdec.cdec_dataset('MSD',t_min,t_max,sensor=25,cache_dir='env_data')
msd_temp['temp']=('time'), (msd_temp['sensor0025']-32)*5./9

##
sjd_flow=common.sjd_flow(np.datetime64("2018-03-01"),
                         np.datetime64("2018-04-20"))

hor_flow=common.oh1_flow(np.datetime64("2018-03-01"),
                         np.datetime64("2018-04-20"))

##

adcp_dir="../../../field/adcp/040518_BT"
ncs=glob.glob(os.path.join(adcp_dir,'*avg_with_time.nc'))

adcp_times=[]
for nc in ncs:
    ds=xr.open_dataset(nc)
    adcp_times.append(ds.time.values)
    ds.close()
##

# Time of arrival vs. flow conditions
fig=plt.figure(1)
fig.clf()
fig.set_size_inches((7.0,4.25),forward=True)

fig,(ax_tag,ax_Q,ax_temp)=plt.subplots(3,1,sharex=True,num=1)

times=utils.unix_to_dt64(df_start.t_mid.values)
t=np.linspace(df_start.t_mid.min(),df_start.t_mid.max(),400)

t_sort=np.sort(times)
for i,rel_date in enumerate(tag_releases):
    lbl='__nolabel__'
    if i==0:
        name=' Upper Tag\n Release'
    else:
        name=' Lower Tag\n Release'
    l=ax_tag.axvline(rel_date,label=lbl,ymax=0.65)
    ax_tag.text(rel_date,0.65,name,fontstyle='italic',va='top',
                fontsize=9,
                transform=ax_tag.get_xaxis_transform())
    
ax_tag.plot(np.r_[t_min,t_sort],np.arange(len(times)+1),'k-',
            label='Tag Arrivals')

if 0: # exploring timing of the two releases arriving.
    for df,label in zip([df_upper,df_lower],['Upper','Lower']):
        times=utils.unix_to_dt64(df.t_mid.values)
        t_sort=np.sort(times)
        ax_tag.plot(np.r_[t_min,t_sort],np.arange(len(times)+1),'k--',
                    label=f'{label} Tag Arrivals')
    
ax_tag.set_ylabel('Tag count')

ax_Q.plot(msd_flow.time,msd_flow.flow_m3s,
          label='MSD Flow')
ax_Q.plot(sjd_flow.time,sjd_flow.flow_m3s,
          label='SJD Flow')
ax_Q.plot(hor_flow.time,hor_flow.flow_m3s,
          label='OH1 Flow')
ax_Q.axhline(0,color='k',lw=0.6,zorder=-2)

for ti,t in enumerate(adcp_times):
    if ti==0:
        lbl='ADCP\nTransects'
    else:
        lbl='__nolabel__'
    ax_Q.axvline(t,color='0.65',label=lbl,zorder=-1)
ax_Q.set_ylabel('Flow (m$^3$ s$^{-1}$)')

ax_temp.plot(msd_temp.time,msd_temp.temp,label='Temperature')
ax_turb=ax_temp.twinx()
turb_col='orange'
temp_col=ax_temp.lines[0].get_color()

ax_turb.plot(msd_turb.time,msd_turb.turb_lp,label='Turbidity',color=turb_col)
ax_turb.set_ylabel('NTU') # ,color=turb_col)
ax_temp.set_ylabel(r'$^{\circ}$C')# ,color=temp_col)

#plt.setp(ax_turb.get_yticklabels(),color=turb_col)
#plt.setp(ax_temp.get_yticklabels(), color=temp_col)

fig.autofmt_xdate()
fig.subplots_adjust(left=0.13,top=0.98,bottom=0.17,right=0.70)
fig.align_ylabels()

ax_tag.axis(xmin=t_min, xmax=t_max)

ax_tag.legend(loc='upper left',bbox_to_anchor=[1.13,1],frameon=0)
ax_Q.legend(loc='upper left',bbox_to_anchor=[1.13,1],frameon=0)
ax_temp.legend(handles=ax_temp.lines+ax_turb.lines,loc='upper left',bbox_to_anchor=[1.13,1],
               frameon=0)

# Panel labels:
for ax,letter in [(ax_tag,'(a)'),
                  (ax_Q,'(b)'),
                  (ax_temp,'(c)')]:
    ax.text(0.015,0.95,letter,va='top',transform=ax.transAxes)

##
fig.savefig('fig-timeline.png',dpi=300)

