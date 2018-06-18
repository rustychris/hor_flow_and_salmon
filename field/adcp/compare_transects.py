# Read the collection of csv files associated with a sontek transect
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import glob

import read_sontek
from stompy.memoize import memoize

##

fig_dir="figs-20180509"
os.path.exists(fig_dir) or os.mkdir(fig_dir)

##

bt_dir='040518_7_BTref'
gps_dir='040518_7_GPSref'
rivr_fns=[os.path.basename(f) for f in glob.glob('%s/*.rivr'%bt_dir)] 

all_bt=[]
all_gps=[]

for fn in rivr_fns:
    ds=read_sontek.surveyor_to_xr('%s/%s'%(bt_dir,fn),
                                  proj='EPSG:26910')
    all_bt.append(ds)

    ds=read_sontek.surveyor_to_xr('%s/%s'%(gps_dir,fn),
                                      proj='EPSG:26910')
    all_gps.append(ds)
    
##

@memoize()
def bathy():
    from stompy.spatial import field
    return field.GdalGrid('../bathy/OldRvr_at_SanJoaquinRvr2012_0104-utm-m_NAVD.tif')

## 
# Plan view, scatter and quiver

for repeat,(ds_bt,ds_gps) in enumerate(zip(all_bt, all_gps)):
    plt.figure(2).clf()
    fig,ax=plt.subplots(num=2)
    
    print repeat
    for i,(ds,col) in enumerate( zip( [ds_bt,ds_gps],
                                      ['g','b'] )):
        avg_east=ds.Ve.mean(dim='cell')
        avg_north=ds.Vn.mean(dim='cell')
        quiv=ax.quiver(ds.x_utm.values, ds.y_utm.values, avg_east.values, avg_north.values,
                       color=col,width=0.0005)
        ax.text( 0.05, 0.95-0.04*i,ds.rivr_filename,
                 transform=ax.transAxes,color=col)

    ax.axis('equal')
    ax.axis( (647145., 647305., 4185815., 4185922.) )

    bathy().plot(ax=ax,cmap='gray',vmin=-20,vmax=6)

    fig.savefig(os.path.join(fig_dir,'quiver-compare-%02d.png'%repeat))

##
for repeat,(ds_bt,ds_gps) in enumerate(zip(all_bt, all_gps)):
    # Scatter of the depth-averaged velocities
    plt.figure(3).clf()
    fig,ax=plt.subplots(num=3)

    ax.plot( ds_bt.Ve.mean(dim='cell'),
             ds_gps.Ve.mean(dim='cell'),
             'r.',label='V east')
    ax.plot( ds_bt.Vn.mean(dim='cell'),
             ds_gps.Vn.mean(dim='cell'),
             'k.',label='V north')
    ax.set_xlabel('BT')
    ax.set_ylabel('GPS')

    ax.axis('equal')
    ax.legend()

    ax.plot([-1,1],
            [-1,1],
            'b-',lw=0.5,label='__nolabel__')

    ax.text(0.95,0.05,"%s\n%s"%(ds_bt.rivr_filename,ds_gps.rivr_filename),
            ha='right',
            transform=ax.transAxes)

    fig.savefig(os.path.join(fig_dir,'scatter-compare-%02d.png'%repeat))

##

# Hope/verify that vertical structure is unaffected.

plt.figure(4).clf()
fig,ax=plt.subplots(num=4)

ax.plot( (ds_bt.Ve - ds_bt.Ve.mean(dim='cell')).values.ravel(),
         (ds_gps.Ve - ds_gps.Ve.mean(dim='cell')).values.ravel(),
         'r.',label='V east')
ax.plot( (ds_bt.Vn - ds_bt.Vn.mean(dim='cell')).values.ravel(),
         (ds_gps.Vn - ds_gps.Vn.mean(dim='cell')).values.ravel(),
         'k.',label='V north')
ax.set_xlabel('BT')
ax.set_ylabel('GPS')

ax.axis('equal')
ax.legend()

ax.plot([-1,1],
        [-1,1],
        'b-',lw=0.5,label='__nolabel__')

fig.savefig(os.path.join(fig_dir,'scatter3d-compare-%02d.png'%repeat))

##

# How much noise in the boat speed?
plt.figure(5).clf()
fig,ax=plt.subplots(num=5)

ax.hist( ds_bt.boat_speed,bins=50,color='g',histtype='step' )
ax.hist( ds_gps.boat_speed,bins=50,color='b',histtype='step' )

##

# How much noise in the boat acceleration?
plt.figure(5).clf()
fig,ax=plt.subplots(num=5)

from scipy import stats
kernel_bt = stats.gaussian_kde(np.diff(ds_bt.boat_speed.values))
kernel_gps= stats.gaussian_kde(np.diff(ds_gps.boat_speed.values))

deltas=np.linspace(-0.5,0.5,100)    

ax.plot(deltas,kernel_bt(deltas),'g-',label='BT')
ax.plot(deltas,kernel_gps(deltas),'b-',label='GPS')

ax.legend()

##

# All tracks, both references
plt.figure(6).clf()
fig,ax=plt.subplots(num=6)

# How well do the transects line up with each other?
for i,(ds_bt,ds_gps) in enumerate(zip(all_bt,all_gps)):
    if i==0:
        bt_label='BT'
        gps_label='GPS'
    else:
        bt_label=gps_label='__nolabel__'
    ax.plot( ds_bt.x_utm,ds_bt.y_utm,'g-',lw=0.7,label=bt_label)
    ax.plot( ds_gps.x_utm,ds_gps.y_utm,'b-',lw=0.7,label=gps_label)
ax.legend()

bathy().plot(ax=ax,cmap='gray',vmin=-20,vmax=6)

zoom=(647148.548017522, 647297.5963927523, 4185810.3082341366, 4185932.5279018255)
ax.axis(zoom)

fig.savefig(os.path.join(fig_dir,'all-transects-path.png'))


##

# All tracks, split panels
plt.figure(7).clf()
fig,axs=plt.subplots(1,2,sharex=True,sharey=True,num=7)
fig.set_size_inches( (10,5), forward=True )

from matplotlib import cm
# How well do the transects line up with each other?
for i,(ds_bt,ds_gps) in enumerate(zip(all_bt,all_gps)):
    col=cm.jet(i/(len(all_bt)-1.0))

    if i==0:
        bt_label='BT'
        gps_label='GPS'
    else:
        bt_label=gps_label='__nolabel__'
        
    axs[0].plot( ds_bt.x_utm, ds_bt.y_utm,color=col,lw=1.0,label=bt_label)
    axs[1].plot( ds_gps.x_utm,ds_gps.y_utm,color=col,lw=1.0,label=gps_label)
axs[0].legend()
axs[1].legend()

for ax in axs:
    bathy().plot(ax=ax,cmap='gray',vmin=-20,vmax=6)

    zoom=(647148.548017522, 647297.5963927523, 4185810.3082341366, 4185932.5279018255)
    ax.axis(zoom)

fig.savefig(os.path.join(fig_dir,'all_paths-2panels.png'))

##

# The immediate questions:
#  - Is BT or GPS more reliable for getting water velocity?
#     location and depth-averaged velocity have some hiccups in the GPS data.
#     see quiver-compare-00.png
#     The differences, as expected, are confined to depth-averaged velocity.
#     Vertical structure is identical.
#     Check second repeat: BT is a bit jerky here, and GPS does not suffer
#       noticeable jumps.
#     Third repeat: very similar
#     Histogram of boat speed is not particularly informative.
#     Histogram of acceleration is mixed, too, slightly suggests GPS is noisier.
#     All paths together: GPS overall is noisier.  There could be some filtering in
#      the BT data, but seems unlikely.

#  - Are the repeat transects "sufficiently" aligned?
#      - ultimately, is there good correlation between them
#      - and is there anything that can be improved in future collection?

#  - Are there other fields which could be extracted to get backscatter?
