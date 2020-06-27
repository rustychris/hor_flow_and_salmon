import pandas as pd
import numpy as np

df_tags=pd.read_csv('cleaned_df.csv',
                    parse_dates=["dt"])

##

# df_tags.dt.max()
# Out[1100]: Timestamp('2018-04-11 14:07:17')
# 
# In [1101]: df_tags.dt.min()
# Out[1101]: Timestamp('2018-03-17 01:03:59')

## 

# Get SJ flows around then, at Mossdale

# mossdale=11304200
# from stompy.io.local import usgs_nwis
# ds=usgs_nwis.nwis_dataset(station=mossdale,
#                           start_date=np.datetime64("2018-03-15"),
#                           end_date=np.datetime64("2018-04-15"),
#                           products=[, days_per_request='M', frequency='realtime', cache_dir=None, clip=True, cache_only=False)
# 

from stompy.io.local import cdec

start_date=np.datetime64("2018-03-15")
end_date=np.datetime64("2018-04-15")
ds=cdec.cdec_dataset(station="MSD",
                     start_date=start_date,
                     end_date=end_date,
                     sensor=20,cache_dir='.')
##

fig=plt.figure(1)
fig.set_size_inches([8,6],forward=True)
fig.clf()

ax2=plt.gca()
ax=ax2.twinx()

ls_flow=ax.plot(ds.time,ds.sensor0020,label="MSD Flow")[0]

ax.set_ylabel('Mossdale (CFS)')
col=ls_flow.get_color()
ax.yaxis.label.set_color(col)
plt.setp(ax.get_yticklabels(),color=col)

from stompy import utils

bins=np.arange(start_date,end_date,np.timedelta64(1,'D'))

col='darkorange'
ax2.hist(utils.to_dnum(df_tags.dt.values),
         bins=utils.to_dnum(bins),color=col)
ax2.yaxis.label.set_color(col)
plt.setp(ax2.get_yticklabels(),color=col)
ax2.set_ylabel('Detections')

Qmodel=220 * 35.314667
time_model=np.datetime64("2018-04-05")
# ax.axhline(Qmodel,color='k',lw=0.5,zorder=-2)
ax.axvline(time_model,color='k',lw=0.5,zorder=-10)

ax.annotate(" ADCP data\n collection",[time_model,Qmodel],
            ha='left',
            va='top' )

##

fig.savefig('tags_and_Qmossdale.png')
