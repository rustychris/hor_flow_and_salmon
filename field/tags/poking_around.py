import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from stompy.spatial import field
from stompy import utils
##
df=pd.read_csv("cleaned_half meter.csv",parse_dates=['dt'])

##

dem=field.GdalGrid('../../bathy/junction-composite-dem-no_adcp.tif')
zoom=(647100, 647600, 4185628, 4185963)
dem_clip=dem.crop(zoom)

NOFLAG=0
FLAG_LAND=1
FLAG_DVEL=2

##

def get_tag_df(tag):
    tag_df=df[ df.TagID==tag ]
    tag_xy=np.c_[ tag_df.X_UTM.values, tag_df.Y_UTM.values ]

    codes=np.zeros( len(tag_df), np.int32)
    codes[:]=NOFLAG

    # Filter land positions
    z=dem_clip(tag_xy)
    # note that this fails when the position is on the other side of a levee
    # there is no check that the transition crossed land.
    codes[z>2.5] = FLAG_LAND

    time_s=tag_df.Epoch_Sec
    # this will bet updated as points are removed, and removed
    # points will keep the dvel that got them ejected.
    point_dvel=np.zeros(len(tag_df),np.float64)

    thresh=2.0 # change in speed in m/s

    while np.sum(codes==0)>1:
        # narrow to the valid set 
        sel=(codes==NOFLAG) # valid mask
        sel_seg_uv=np.diff(tag_xy[sel],axis=0) / np.diff(time_s[sel])[:,None]
        sel_dvel=utils.mag( np.diff(sel_seg_uv,axis=0) )
        sel_dvel=np.concatenate( ( [0],sel_dvel,[0] ) ) # pad to ends.
        sel_worst=np.argmax(sel_dvel)
        if sel_dvel[sel_worst]<thresh:
            break
        # map back to original index
        orig_idx=np.nonzero(sel)[0]
        # slices help deal with short datasets
        point_dvel[orig_idx[1:-1]]=sel_dvel[1:-1]
        worst=orig_idx[sel_worst]
        codes[worst]=FLAG_DVEL
    tag_df=tag_df.copy()
    tag_df['code']=codes
    tag_df['point_dvel']=point_dvel
    tag_df['dem_z']=z
    return tag_df

def tag_df_figure(tag_df):
    fig=plt.figure(1)
    fig.clf()
    ax=fig.add_subplot(1,1,1)
    ax.set_title(tag_df.TagID.values[0])

    tag_xy=np.c_[ tag_df.X_UTM.values, tag_df.Y_UTM.values ]
    codes=tag_df.code.values

    # node rendering:
    if 1: # validity codes
        boundaries=[-0.5,0.5,1.5,2.5]
        scat=ax.scatter(tag_xy[:,0],tag_xy[:,1],30,codes,cmap='inferno',
                        vmin=boundaries[0],vmax=boundaries[-1])
        cbar=plt.colorbar(scat,ax=ax,label='Code',ticks=[0,1,2],boundaries=boundaries)
        # values=[0,1,2],)
        cbar.set_ticklabels(["Good","Land","DVel"])
    if 0:
        ax.scatter(tag_xy[:,0],tag_xy[:,1],30,point_dvel)

    # line rendering:
    if 1: # raw detections path
        ax.plot(tag_xy[:,0],tag_xy[:,1],'g-',lw=0.9,alpha=0.4)
    if 1: # filtered path
        ax.plot(tag_xy[codes==0,0],
                tag_xy[codes==0,1],'m-',lw=1.5,alpha=0.9)
        
    if 0: # plot per-segment velocity magnitude
        segs=np.array( [tag_xy[:-1,:], tag_xy[1:,:] ] ).transpose(1,0,2)
        scoll=collections.LineCollection(segs,
                                         array=utils.mag(seg_uv),
                                         cmap='jet') 
        ax.add_collection(scoll)

    dem_clip.plot(ax=ax,cmap='Blues_r')
    ax.axis('equal')
    return fig


tag_ids=df.TagID.unique()

## 
#for tag_id in tag_ids:
for tag_id in ['74dd']:
    tag_df=get_tag_df(tag_id)
    tag_df_figure(tag_df)
    break
#plt.pause(1.0)

