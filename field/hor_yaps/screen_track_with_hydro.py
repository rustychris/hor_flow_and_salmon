"""
Additional screening based on tracks after hydro data has been added.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import track_common
##

input_path="cleaned_v00"
fig_dir=os.path.join(input_path,'figs-20200326')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

df=track_common.read_from_folder(input_path)

##     PLOTS    ##

from scipy.stats import gaussian_kde
from matplotlib import gridspec, collections, widgets
from stompy.spatial import field
import seaborn as sns
import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')

## 
dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")

##

# Concatenate into a single dataframe for some of these plots
for idx,row in df_screen.iterrows():
    row['track']['id'] = row['id']
    
seg_tracks=pd.concat( [ track.iloc[:-1,:]
                        for track in df_screen['track'].values ] )

## 
# Global histogram, un-normalized, geographic coordinates
plt.figure(10).clf()
fig,ax=plt.subplots(1,1,num=10)

bins=np.linspace(-1.,1.,100)

ax.hist2d( seg_tracks['ground_u'].values,
           seg_tracks['ground_v'].values,
           bins=bins,cmap='CMRmap_r')
ax.set_xlabel('East-West')
ax.set_ylabel('North-South')
ax.axis('equal')
ax.set_title("All segments, no normalization")

##

def set_direction_labels(ax):
    ax.set_ylabel('Swim speed (m/s)')
    ax.set_xlabel('Swim speed (m/s)')

    bbox=dict(facecolor='w',lw=0.5)
    common=dict(bbox=bbox,weight='bold',transform=ax.transAxes)

    ax.text( 0.5, 1.0, 'Downstream',va='top',ha='center', **common)
    ax.text( 0.5, 0.0, 'Upstream',  va='bottom',ha='center', **common)
    ax.text( 0.0, 0.5, 'River\nLeft',va='center',ha='left',**common)
    ax.text( 1.0, 0.5, 'River\nRight',  va='center',ha='right',**common)


bandwidth=0.25 # pretty close to what seaborn used in the first place.

# Global histogram, un-normalized, river coordinates
def figure_swim_density(bandwidth=0.25,num=11,
                        weights=None,
                        seg_tracks=seg_tracks,
                        label=""):
    plt.figure(num).clf()
    fig,ax=plt.subplots(1,1,num=num)

    # put u on the vertical to make it more intuitive relative
    # to river.
    # swap signs to keep right-handed coordinate system
    xvals=-seg_tracks['swim_vrel'].values
    yvals= seg_tracks['swim_urel'].values
    
    ax.scatter(xvals, yvals,
               0.2,color='k',alpha=0.4,zorder=3)
    ax.set_xlabel('left  ferry  right')
    ax.set_ylabel('+rheo — -rheo')
    ax.axis('equal')
    ax.set_title(label)

    swim_rel_vec=np.stack( (xvals,yvals) )
    kernel=gaussian_kde(swim_rel_vec,bw_method=bandwidth,weights=weights)

    scale=0.5
    X,Y = np.mgrid[-scale:scale:100j, -scale:scale:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z=np.reshape(kernel(positions).T, X.shape)
    ax.contour(X,Y,Z,10)

    lim=0.5
    ax.axis( [-lim,lim,-lim,lim] )
    ax.axhline(0,color='0.5',lw=0.5)
    ax.axvline(0,color='0.5',lw=0.5)

    set_direction_labels(ax)
    return fig

fig=figure_swim_density(label="All segments, un-normalized",num=11)
fig.savefig(os.path.join(fig_dir,"all_segments-unnormalized.png"))

##

# Weights -- segments per individual sum to unit weight
seg_counts=seg_tracks.groupby(seg_tracks.id).size()
weights=1./seg_counts[ seg_tracks.id]

fig=figure_swim_density(label="All segments, weights=1/<count per individual>",
                        weights=weights,num=12)
fig.savefig(os.path.join(fig_dir,"all_segments-weight_by_individual.png"))

##

weights=(seg_tracks.tnum_end - seg_tracks.tnum)

# Global histogram, normalized per time, river coordinates
fig=figure_swim_density(label="All segments, weights=<segment $\Delta$t>",
                        weights=weights,num=13)
fig.savefig(os.path.join(fig_dir,"all_segments-weight_by_time.png"))

##

def fig_track_swimming(seg_track,num=20,zoom=None,buttons=True):
    seg_track=seg_track[ np.isfinite(seg_track['ground_u'].values) ]
    
    fig=plt.figure(num)
    fig.set_size_inches((8,9),forward=True)
    fig.clf()
    gs=gridspec.GridSpec(2,2)

    ax=fig.add_subplot(gs[0,0])

    segs=np.array([ seg_track.loc[ :, ['x','y'] ].values,
                    seg_track.loc[ :, ['x_end','y_end'] ].values ]).transpose(1,0,2)

    elapsed=seg_track.index.values - seg_track.index.values[0]

    if zoom is None:
        pad=50.0
        zoom=[ segs[...,0].min() - pad,
               segs[...,0].max() + pad,
               segs[...,1].min() - pad,
               segs[...,1].max() + pad ]
    seg_coll=collections.LineCollection( segs,array=elapsed,cmap=turbo)
    ax.add_collection(seg_coll)
    ax.axis('equal')

    plots={}
    
    # Hydrodynamic surface velocity
    plots['hydro_quiver']=ax.quiver( seg_track.x_m, seg_track.y_m,
                                     seg_track[model_u], seg_track[model_v],
                                     color='k')

    plots['groundspeed_quiver']=ax.quiver( seg_track.x_m, seg_track.y_m,
                                           seg_track.ground_u, seg_track.ground_v,
                                           color='orange')

    plots['swim_quiver']=ax.quiver( seg_track.x_m, seg_track.y_m,
                                    seg_track.swim_u, seg_track.swim_v,
                                    color='b')
    plots['togglers']=[plots['hydro_quiver'],
                       plots['groundspeed_quiver'],
                       plots['swim_quiver']]

    if buttons:
        button_ax=fig.add_axes([0.01,0.8,0.2,0.12])
        button_ax.axis('off')
        plots['button_ax']=button_ax
        plots['buttons']=widgets.CheckButtons( button_ax,["Hydro","Smooth","Swim"])

        def button_cb(arg,plots=plots):
            # Unclear why this has to be negated
            [ t.set_visible(stat)
              for t,stat in zip( plots['togglers'],
                                 plots['buttons'].get_status())]
            plots['fig'].canvas.draw()

        plots['buttons'].on_clicked(button_cb)
        
    for t in plots['togglers']:
        t.set_visible(0)
    
    dc=dem.crop([ zoom[0]-500,
                  zoom[1]+500,
                  zoom[2]-500,
                  zoom[3]+500] )
    dc.plot(ax=ax,zorder=-5,cmap='gray',clim=[-20,10],interpolation='bilinear')
    dc.plot_hillshade(ax=ax,z_factor=1,plot_args=dict(interpolation='bilinear'))
    ax.axis(zoom)
    ax.axis('off')

    ax_hist=fig.add_subplot(gs[1,:])

    xvals=-seg_track['swim_vrel'] 
    yvals=seg_track['swim_urel']
    sns.kdeplot(xvals,yvals,
                shade_lowest=False,
                clip=[ [-0.5,0.5],[-0.5,0.5]],
                linewidths=0.5,
                ax=ax_hist)

    scat=ax_hist.scatter( xvals,yvals, 8, elapsed,cmap=turbo )
    ax_hist.set_xlabel('left  ferry  right')
    ax_hist.set_ylabel('+rheo — -rheo')
    ax_hist.axis('equal')
    ax_hist.axhline(0.0,color='0.5',lw=0.5)
    ax_hist.axvline(0.0,color='0.5',lw=0.5)
    plt.colorbar(scat,ax=ax_hist,label="Elapsed time (s)")

    Uscale=0.4
    ax_hist.axis( [-Uscale,Uscale,-Uscale,Uscale] )
    fig.tight_layout()
    plots['fig']=fig

    set_direction_labels(ax_hist)
    
    # Text info
    ax_txt=fig.add_subplot(gs[0,1])
    ax_txt.axis('off')
    plots['tag']=seg_track.id.values[0]
    txt="Tag: %s"%(plots['tag'])
    ax_txt.text(0.05,0.95,txt,va='top')

    plots['ax_hist']=ax_hist

    return plots

# for tag in ['7a96','7577']
# 7a96 is picture perfect.  But it has pretty big standard deviations in this run,
# and didn't make the cut.  It still looks okay. So maybe worth relaxing that constraint?
# 7577 is positive rheotaxis
df_by_id=df_screen.set_index('id')

plots=fig_track_swimming(df_by_id.loc['7659','track'],
                         num=20)
## 
tag_fig_dir=os.path.join(fig_dir,'tags')
if not os.path.exists(tag_fig_dir):
    os.makedirs(tag_fig_dir)
    
for tag in df_by_id.index.values:
    print(tag)
    plots=fig_track_swimming(df_by_id.loc[tag,'track'],num=21,
                             buttons=False)
    plots['fig'].savefig(os.path.join(tag_fig_dir,'%s-swimming.png'%plots['tag']))

##

# High count tracks in the previous data.
# check to see if we've lost to many tracks

# 7275     80
# 7a95     83
# 7b2b     83
# 7b65     83
# 7499     98
# 7a96    101
# 7659    106
# 768a    108
# 7d49    135
# 7ca5    156
# 75d5    193
# 7577    248
