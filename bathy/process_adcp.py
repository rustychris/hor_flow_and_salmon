"""
Ingest ADCP data, eventually with concurrent RTK data, and
develop merged DEM.
"""
from stompy import utils
from stompy.plot import plot_utils
import stompy.plot.cmap as scmap
from stompy.spatial import proj_utils

import glob
import pandas as pd
import re
import bathy

utils.path("../field/adcp")
import read_sontek

##

patts=[
    "../field/adcp/bathy_data/*.riv",
    "../field/adcp/040518_BT/*/*.rivr",
    "../field/adcp/040518_BT/*/*.riv",
]

rivr_fns=[]
for patt in patts:
    rivr_fns.extend( glob.glob(patt) )

##

gages=[]

mossdale="B95820"
sj_below_hor="B95765"

wdl_codes=[mossdale,sj_below_hor]
for wdl_code in wdl_codes:
    url="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/%s/2018/STAGE_15-MINUTE_DATA_DATA.CSV"%wdl_code
    loc_fn="derived/wdl-%s-stage-wy2018.csv"%wdl_code
    if not os.path.exists(loc_fn):
        print("Fetching to %s"%loc_fn)
        utils.download_url(url,loc_fn)

    df=pd.read_csv(loc_fn,
                   names=['time','stage','qual','comment'],
                   header=None,skiprows=3,parse_dates=['time'],infer_datetime_format=True)

    loc=df.comment[1]

    lat=float(re.search(r'Lat:([-0-9.]+)',loc).group(1))
    lon=float(re.search(r'Long?:([-0-9.]+)',loc).group(1))

    gage=xr.Dataset.from_dataframe(df.set_index('time'))
    gage.attrs['url']=url
    gage.attrs['src']="Water Data Library"
    gage.attrs['code']=wdl_code
    gage['lat']=(),lat
    gage['lon']=(),lon
    ll=[lon,lat]
    xy=proj_utils.mapper('WGS84','EPSG:26910')(ll)
    gage['x']=(),xy[0]
    gage['y']=(),xy[1]

    gages.append(gage)

##

# How do those time series compare?  Quick glance:
if 0:
    plt.figure(2).clf()
    fig,ax=plt.subplots(1,1,num=2)

    for gage in gages:
        ax.plot(gage.time,gage.stage,label=gage.attrs['code'])
    ax.legend()

# What is the time zone on those? assume PST?
# What is the time zone in the ADCP data?  appears to be PST-0100, or UTC-0900

##

# First go, just use the bottom track depth, not the per-beam data.
# Assemble a set of xyz from all above data.
# will add in progressive complexity:
#  1. all relative to transducer
#  2. Offset for transducer depth below water
#  3. Timeseries of water surface elevation
#  4. Multiple beams?

all_xyz=[]
all_t=[]

surface_to_transducer=0.20 # [m]

for rivr_fn in rivr_fns:
    print(rivr_fn)
    ds=read_sontek.surveyor_to_xr(rivr_fn,proj="EPSG:26910")

    x=ds.x_sample.values
    y=ds.y_sample.values
    z=-ds.depth_bt.values # positive up, relative to transducer
    z-=surface_to_transducer
    all_xyz.append( np.c_[x,y,z] )

    all_t.append(ds.time.values)

xyz_samples=np.concatenate(all_xyz)
t_samples=np.concatenate(all_t)

##

# Assume that the ADCP is in PST (that's probably an hour wrong, but
# need to check with Mike).

if 1: # Attempt to adjust to NAVD88.
    # Not sure that these gages even purport to be NAVD88
    # roughly speaking, call the HOR halfway between the two gages.
    gages_at_t=[]

    for gage in gages:
        t_interp=np.interp( utils.to_dnum(t_samples),
                            utils.to_dnum(gage.time.values), gage.stage.values,
                            left=np.nan,right=np.nan)
        gages_at_t.append(t_interp)

    stage_ft=0.5*(gages_at_t[0] + gages_at_t[1])
    stage_m=0.3048 * stage_ft

    xyz_samples[:,2] += stage_m

##

# These range 8-14.  That sounds like PST or PDT.
# ideally would just get the real answer from the GPS
# clock.  For the first of the transect 2 repeats,
# GPS['Utc'] reports 174336.5, incrementing a second
# each sample.  That's 17:43:36.500
#
# hours = xr.DataArray(t_samples).dt.hour.values
# plt.figure(3).clf() ; plt.hist(hours)

# Looking at a specific transect with matlab info:
# '../field/adcp/040518_BT/040518_2BTref/20180404084019r.rivr'
# '../field/adcp/040518_BT/040518_2BTref/20180404084019r.mat'
# First time in the dataset is '2018-04-04T08:40:10.000000000'
#  with 167 samples
# First time in the GPS['Utc'] field is
# [174336.5], also 167 samples.
# Kind of weird, but maybe some clock drift?
#   That is 17:43:36.500 vs 08:40:10.00
#   So the ADCP is 9h03 behind.  That's weird, since PDT is 7h off,
#   and PST is 8h off.
#
##

zoom=(646982, 647582, 4185480, 4186091)
dem=bathy.dem().extract_tile(zoom,res=2.0)

##

delta=xyz_samples[:,2] - dem(xyz_samples[:,:2])

##
plt.figure(1).clf()
fig,(ax,ax2)=plt.subplots(1,2,num=1,sharex=True,sharey=True)
fig.set_size_inches([10,7],forward=True)

scat=ax.scatter( xyz_samples[:,0],xyz_samples[:,1],
                 15, xyz_samples[:,2], cmap='jet')

plot_utils.cbar(scat,label='m NAVD88',ax=ax,orientation='horizontal')

img=dem.plot(ax=ax,interpolation='nearest',cmap='jet')

crange=[-6,2.25]
scat.set_clim(crange)
img.set_clim(crange)

div_cmap=scmap.load_gradient('ViBlGrWhYeOrRe.cpt')

# And show the difference:
scat=ax2.scatter( xyz_samples[:,0],xyz_samples[:,1],
                  15, delta, cmap='seismic')
plot_utils.cbar(scat,label='ADCP - DEM (m)',ax=ax2,orientation='horizontal')

img=dem.plot(ax=ax2,interpolation='nearest',cmap='gray')

scat.set_clim([-1,1])
img.set_clim([-10,5])

for a in [ax,ax2]:
    a.xaxis.set_visible(0)
    a.yaxis.set_visible(0)

fig.tight_layout()

fig.savefig('dem-vs-adcp-v01.png',dpi=125)

# Save some details
ax.axis( (647365.2402763385, 647536.919447323, 4185540.790668355, 4185717.8615734414) )
fig.savefig('dem-vs-adcp-v01-zoom_upstream.png',dpi=125)
ax.axis( (647120.3645595856, 647334.6308117444, 4185737.7525997157, 4185958.7480704053) )
fig.savefig('dem-vs-adcp-v01-zoom_junction.png',dpi=125)
ax.axis( (647174.8171692798, 647325.0441959006, 4185850.7106188717, 4186005.6556562902) )
fig.savefig('dem-vs-adcp-v01-zoom_scour.png',dpi=125)
ax.axis( (647092.4543586917, 647242.6813853125, 4185760.37721371, 4185915.3222511285) )
fig.savefig('dem-vs-adcp-v01-zoom_barrier.png',dpi=125)


##

from stompy.spatial import proj_utils, wkb2shp
from shapely import geometry

ll=proj_utils.mapper('EPSG:26910','WGS84')(xyz_samples[:,:2])


# Write the points so far to a shapefile, which can then be translated to kml
geoms=[ geometry.Point(*pnt) for pnt in ll]


wkb2shp.wkb2shp('samples-depth.shp',geoms,srs_text='WGS84',
                fields={'depth':xyz_samples[:,2]},overwrite=True)
