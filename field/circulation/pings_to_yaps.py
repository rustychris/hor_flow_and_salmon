# reformat ping data for yaps (at least as much as is easier on the python side).
import xarray as xr
import pandas as pd
## 
pings=xr.open_dataset('pings-2018-03-17T16:08:23.763998000_2018-03-17T18:05:33.221453000.nc')

##

# What are the mean offsets between rxs?
matrix=pings.matrix.values

# these are all fairly small -- less than 4s.
for a in range(matrix.shape[1]):
    for b in range(a+1,matrix.shape[1]):
        offset=np.nanmean( matrix[:,a] - matrix[:,b])
        print(f"{a} -- {b}: {offset}")

## 

# the sample data has
# Date and Time (UTC),Receiver,Transmitter,Transmitter Name,Transmitter Serial,Sensor Value,Sensor Unit,Station Name,Latitude,Longitude
# 2019-09-09 16:04:11.193,VR2W-128355,A69-1602-59335,,,,,CESI10

# Seems that no records have lat/lon
# no transmitter name, tx serial, sensor value, sensor unit.
# but date and time (with ms fraction) are there, along with rx and tx, and station name.


## generate $detections for beacon tags.

beacon_tags=pings.rx_beacon.values

is_beacon=np.array( [ t in beacon_tags
                      for t in pings.tag.values ] )
is_multirx=(np.isfinite(pings.matrix).sum(axis=1)>1).values
is_triplerx=(np.isfinite(pings.matrix).sum(axis=1)>2).values



##
bpings=pings.isel(index=is_beacon&is_triplerx)

b0=bpings.to_dataframe()

##
b1=b0[ ~b0.matrix.isnull() ].reset_index().loc[:, ['matrix','rx','tag']]
b1['epo']=np.floor(b1.matrix)
b1['frac']=b1.matrix - b1.epo
b2=b1.rename(columns={'rx':'serial'})
del b2['matrix']
# everything but ts, which is just a text timestamp.

b2.to_csv('../hor_yaps/detections.csv',index=False)
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
sync_tags=df['sync_tag']
good_tags=b2.tag.unique()
df.loc[~sync_tags.isin(good_tags), 'sync_tag']=np.nan
df.to_csv('../hor_yaps/hydros.csv',index=False)

##

# Which rx are the most reliable?
# there were 2 that I think got switched... sm2/sm3
# 
