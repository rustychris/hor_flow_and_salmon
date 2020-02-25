# reformat ping data for yaps (at least as much as is easier on the python side).
import xarray as xr
import pandas as pd
import numpy as np
##
#fn='pings-2018-03-17T16:08:23.763998000_2018-03-17T18:05:33.221453000.nc'
# This includes more padding before/after period of interest
fn='pings-2018-03-17T12:38:23.763998000_2018-03-17T21:35:33.221453000.nc'
pings_orig=xr.open_dataset(fn)

##
# the sample data has
# Date and Time (UTC),Receiver,Transmitter,Transmitter Name,Transmitter Serial,Sensor Value,Sensor Unit,Station Name,Latitude,Longitude
# 2019-09-09 16:04:11.193,VR2W-128355,A69-1602-59335,,,,,CESI10

# Seems that no records have lat/lon
# no transmitter name, tx serial, sensor value, sensor unit.
# but date and time (with ms fraction) are there, along with rx and tx, and station name.


## generate $detections for beacon tags.

# Omit AM3 from the whole thing...
rx_mask=(pings_orig.rx!='AM3')
pings=pings_orig.isel(rx=rx_mask)

beacon_tags=pings.rx_beacon.values

is_beacon=np.array( [ t in beacon_tags
                      for t in pings.tag.values ] )
is_multirx=(np.isfinite(pings.matrix).sum(axis=1)>1).values
is_triplerx=(np.isfinite(pings.matrix).sum(axis=1)>2).values

# less restrictive than triplerx -- limit the above to tags
# that at least once are seen by 3 rx

good_tags=[t # .item()
           for t in np.unique(pings.isel(index=is_triplerx).tag.values)]

is_goodtag=np.array( [ t in good_tags
                       for t in pings.tag.values])

##

for mask,fn in [
        (is_goodtag & is_multirx, 'all_detections'),
        #(is_triplerx, 'all_detections'),
        #(is_goodtag & is_multirx & is_beacon,  'beacon_detections') 
        #(is_triplerx & is_beacon,  'beacon_detections')
]:
    bpings=pings.isel(index=mask)

    b0=bpings.to_dataframe()

    b1=b0[ ~b0.matrix.isnull() ].reset_index().loc[:, ['matrix','rx','tag']]
    b1['epo']=np.floor(b1.matrix)
    b1['frac']=b1.matrix - b1.epo
    b2=b1.rename(columns={'rx':'serial'})
    del b2['matrix']
    # everything but ts, which is just a text timestamp.
    b2.to_csv(f'../hor_yaps/{fn}.csv',index=False)

##

# generate the equivalent of ssu1$hydros
# Try omitting sync_tag for sync_tags that are not heard by at least 3 rxs.
hydros=xr.Dataset()
hydros['serial']=pings.rx
hydros['x']=pings.rx_x
hydros['y']=pings.rx_y
hydros['z']=0*pings.rx_z # this z info is probably useless.
hydros['sync_tag']=pings.rx_beacon
hydros['idx']=('rx',), 1+np.arange(pings.dims['rx'])
df=hydros.to_dataframe()
sync_tags=df['sync_tag'].values
if isinstance(sync_tags[0],np.ndarray):
    # gets weird -- sometimes this is necessary
    df['sync_tag']=[t.item() for t in sync_tags]

# good_tags=b2.tag.unique()
# during testing, this kills FF24
df.loc[~df['sync_tag'].isin(good_tags), 'sync_tag']=np.nan
df.to_csv('../hor_yaps/hydros.csv',index=False)

##

# What kind of mean shift is there?

matrix=pings.matrix.values

max_offset=0.0

for a in range(pings.dims['rx']):
    for b in range(a+1,pings.dims['rx']):
        offset=np.nanmean(matrix[:,b] - matrix[:,a])
        max_offset=max(max_offset, np.abs(offset))
        print(f"{a:2} -> {b:2}  {offset:+6.2f}s")
print(f"Max offset: {max_offset:.2f}s")
    
##

# Compare the results:
import matplotlib.pyplot as plt
from matplotlib import collections
from stompy.spatial import field


tag='7adb'
df_yaps=pd.read_csv("../hor_yaps/track-7adb.csv")
df_segs=pd.read_csv("../../analysis/swimming/code/segments.csv")
df_clean=pd.read_csv("../../analysis/swimming/code/cleaned_half_meter.csv")

df_segs_tag=df_segs[ df_segs.id=='7adb' ]
df_clean_tag=df_clean[ df_clean.ID=='7adb' ]

dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")

## 

fig=plt.figure(1)
fig.clf()
ax=fig.add_axes([0,0,1,1])

seg1=df_segs_tag.loc[:, ['x1','y1']]
seg2=df_segs_tag.loc[:, ['x2','y2']]

segs=np.stack( [seg1,seg2], axis=1)
ax.plot(df_clean_tag.X, df_clean_tag.Y,label='cleaned_half_meter',zorder=1)
ax.add_collection( collections.LineCollection(segs,color='r',label='segments'))
ax.plot( df_yaps.x, df_yaps.y, label='YAPS' )


dem.plot(cmap='gray',ax=ax,clim=[-50,30])
dem.plot(cmap='gray',ax=ax,clim=[-50,30])

zoom=(647076.2999999999, 647511.9, 4185588.45, 4186064.75)

pad=1000
dc=dem.crop([ zoom[0]-pad,
              zoom[1]+pad,
              zoom[2]-pad,
              zoom[3]+pad] )
img=dc.plot(ax=ax,zorder=-5,cmap='gray',clim=[-40,10],interpolation='bilinear')
dc.plot_hillshade(ax=ax,z_factor=1,plot_args=dict(interpolation='bilinear',alpha=0.5))
ax.set_adjustable('datalim')
ax.axis(zoom)
ax.axis('off')

ax.legend()

fig.savefig('7adb-yaps-comparison.png',dpi=200)
