"""
Plot tracks with after obvious bad tracks removed.
"""
import os
import glob
from scipy.signal import medfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import track_common
##

input_path="screen_final"
fig_dir=os.path.join(input_path,'figs-20200502')
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

hydros=pd.read_csv('yap-positions.csv')

## 
dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")

##

# Concatenate into a single dataframe for some of these plots
for idx,row in df.iterrows():
    row['track']['id'] = idx
    
seg_tracks=pd.concat( [ track.iloc[:-1,:]
                        for track in df['track'].values ] )

## 

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
    ax.contour(X,Y,Z,5,colors='m')

    lim=0.5
    ax.axis( [-lim,lim,-lim,lim] )
    ax.axhline(0,color='0.5',lw=0.5)
    ax.axvline(0,color='0.5',lw=0.5)

    track_common.set_direction_labels(ax)
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

# Global histogram, normalized per time, river coordinates
weights=(seg_tracks.tnum_end - seg_tracks.tnum)

fig=figure_swim_density(label="All segments, weights=<segment $\Delta$t>",
                        weights=weights,num=13)
fig.savefig(os.path.join(fig_dir,"all_segments-weight_by_time.png"))

##

def medfilt_weighted(data,N,weights=None):
    quantile=0.5
    if weights is None:
        weights=np.ones(len(data))

    data=np.asarray(data)
    weights=np.asarray(weights)
    
    result=np.zeros_like(data)
    neg=N//2
    pos=N-neg
    for i in range(len(result)):
        # say N is 5
        # when i=0, we get [0,3]
        # when i=10, we get [8,13
        slc=slice( max(0,i-neg),
                   min(i+pos,len(data)) )
        subdata=data[slc]
        # Sort the data
        ind_sorted = np.argsort(subdata)
        sorted_data = subdata[ind_sorted]
        sorted_weights = weights[slc][ind_sorted]
        # Compute the auxiliary arrays
        Sn = np.cumsum(sorted_weights)
        # center those and normalize
        Pn = (Sn-0.5*sorted_weights)/Sn[-1]
        result[i]=np.interp(quantile, Pn, sorted_data)
    return result

# Global relationship between swim speed and hydro speed
def figure_swim_hydro_speed(num=23,
                            bandwith=0.25,
                            seg_tracks=seg_tracks,
                            weights=None,
                            label=""):
    plt.figure(num).clf()
    fig,(ax_mag,ax_u,ax_v,ax_usign)=plt.subplots(1,4,num=num)
    fig.set_size_inches((8,2.75),forward=True)

    swim_speed =np.sqrt( seg_tracks['swim_u'].values**2 + seg_tracks['swim_v'].values**2)
    hydro_speed=np.sqrt( seg_tracks['model_u_surf'].values**2 + seg_tracks['model_v_surf'].values**2)

    for ax,dep in [ (ax_mag,swim_speed),
                    (ax_u, np.abs(seg_tracks['swim_urel'].values)),
                    (ax_v, np.abs(seg_tracks['swim_vrel'].values)),
                    (ax_usign,seg_tracks['swim_urel'].values)]:
        ax.plot( hydro_speed, dep, 'k.',ms=1,alpha=0.2)
        order=np.argsort(hydro_speed)
        N=201

        if weights is not None:
            hyd_med=medfilt_weighted( hydro_speed[order], N,weights=weights)
            dep_med=medfilt_weighted( dep[order], N, weights=weights)
        else:
            hyd_med=medfilt( hydro_speed[order], N)
            dep_med=medfilt( dep[order], N)
            
        ax.plot( hyd_med[N:-N], dep_med[N:-N], 'm-')
            
        ax.set_xlabel('Hydro speed (m/s)')
        if ax!=ax_usign:
            ax.set_aspect(1.0)
            ax.axis(xmin=0,ymin=0,xmax=0.8,ymax=0.8)
        else:
            ax.set_aspect(0.5)
            ax.axis(xmin=0,ymin=-0.8,xmax=0.8,ymax=0.8)
            ax.axhline(0,color='#ffff00')
        
    ax_mag.set_ylabel('Swim speed (m/s)')
    ax_mag.set_title(r'$\sqrt{ lon^2+lat^2 }$')
    ax_u.set_title(r'$|lon|$')
    ax_v.set_title(r'$|lat|$')
    ax_usign.set_title(r'$lon$')
    
    plt.setp(ax_u.get_yticklabels(), visible=0)
    plt.setp(ax_v.get_yticklabels(), visible=0)
    ax_usign.yaxis.tick_right()
    #plt.setp(ax_usign.get_yticklabels(), visible=0)

    fig.subplots_adjust(left=0.085,right=0.92,top=0.97,bottom=0.03)
    return fig

fig=figure_swim_hydro_speed()
fig.savefig(os.path.join(fig_dir,"all_segments-swim_hydro_speed.png"),dpi=200)

##

seg_counts=seg_tracks.groupby(seg_tracks.id).size()
weights=1./seg_counts[ seg_tracks.id]

fig=figure_swim_hydro_speed(weights=weights,num=24)
fig.savefig(os.path.join(fig_dir,"all_segments-swim_hydro_speed-per_tag.png"),dpi=200)

##
model_u='model_u'
model_v='model_v'

def fig_track_swimming(seg_track,num=20,zoom=None,buttons=True,
                       swim_filt=None):
    seg_track=seg_track[ np.isfinite(seg_track['ground_u'].values) ]
    
    fig=plt.figure(num)
    fig.set_size_inches((8,9),forward=True)
    fig.clf()
    gs=gridspec.GridSpec(2,2)

    ax=fig.add_subplot(gs[0,0])

    segs=np.array([ seg_track.loc[ :, ['x','y'] ].values,
                    seg_track.loc[ :, ['x_end','y_end'] ].values ]).transpose(1,0,2)

    elapsed=seg_track.tnum_m.values - seg_track.tnum_m.values[0]

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

    ax.plot(hydros.yap_x,hydros.yap_y,'o',mfc='none',mec='k',zorder=-1)
    
    dc=dem.crop([ zoom[0]-500,
                  zoom[1]+500,
                  zoom[2]-500,
                  zoom[3]+500] )
    dc.plot(ax=ax,zorder=-5,cmap='gray',clim=[-20,10],interpolation='bilinear')
    dc.plot_hillshade(ax=ax,z_factor=1,plot_args=dict(interpolation='bilinear',zorder=-4))
    ax.axis(zoom)
    ax.axis('off')

    ax_hist=fig.add_subplot(gs[1,:])

    Uscale=0.6
    xvals=-seg_track['swim_vrel'] 
    yvals=seg_track['swim_urel']

    if swim_filt is not None:
        xvals=swim_filt(xvals)
        yvals=swim_filt(yvals)
        
    sns.kdeplot(xvals,yvals,
                shade_lowest=False,
                clip=[ [-Uscale,Uscale],[-Uscale,Uscale]],
                linewidths=0.5,levels=5,
                ax=ax_hist)

    scat=ax_hist.scatter( xvals,yvals, 8, elapsed,cmap=turbo )
    ax_hist.set_xlabel('left  ferry  right')
    ax_hist.set_ylabel('+rheo — -rheo')
    ax_hist.axis('equal')
    ax_hist.axhline(0.0,color='0.5',lw=0.5)
    ax_hist.axvline(0.0,color='0.5',lw=0.5)
    plt.colorbar(scat,ax=ax_hist,label="Elapsed time (s)")

    ax_hist.axis( [-Uscale,Uscale,-Uscale,Uscale] )
    fig.tight_layout()
    plots['fig']=fig

    track_common.set_direction_labels(ax_hist)
    
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
from stompy import filters
plots=fig_track_swimming(df.loc['7ACD','track'],
                         num=20,
                         #swim_filt=lambda x: medfilt(x,9)
                         # swim_filt=lambda x: filters.lowpass_fir(x,winsize=13))
                         )

## 
tag_fig_dir=os.path.join(fig_dir,'tags')
if not os.path.exists(tag_fig_dir):
    os.makedirs(tag_fig_dir)
    
for tag in df.index.values:
    print(tag)
    plots=fig_track_swimming(df.loc[tag,'track'],num=21,
                             buttons=False)
    plots['fig'].savefig(os.path.join(tag_fig_dir,'%s-swimming.png'%plots['tag']))

##

# Spatial analysis

zoom=[seg_tracks.x.min(),
      seg_tracks.x.max(),
      seg_tracks.y.min(),
      seg_tracks.y.max()]
dc=dem.crop([ zoom[0]-500,
              zoom[1]+500,
              zoom[2]-500,
              zoom[3]+500] )

##
fig=plt.figure(50)
fig.clf()
ax=fig.add_subplot(1,1,1)
ax.set_position([0,0,1,1])
dc.plot(ax=ax,zorder=-5,cmap='gray',clim=[-20,10],interpolation='bilinear')
dc.plot_hillshade(ax=ax,z_factor=1,plot_args=dict(interpolation='bilinear',zorder=-4))

ax.plot( seg_tracks.x, seg_tracks.y, 'r.',alpha=0.4,ms=3)

ax.set_adjustable('datalim')
ax.axis(zoom)
ax.axis('off')
fig.savefig(os.path.join(fig_dir,'all_positions.png'))

##

from stompy.grid import unstructured_grid
g=unstructured_grid.UnstructuredGrid.from_ugrid('../../model/grid/snubby_junction/snubby-01.nc')

# NB: This only grabs the start of each segment
pos_cells=[g.select_cells_nearest(pnt,inside=True)
           for pnt in np.c_[seg_tracks.x,seg_tracks.y]]

## 
gwet=g.copy()
gwet.add_cell_field('cell_depth',g.cells['cell_depth'])

for c in np.nonzero( gwet.cells['cell_depth']>1.9 )[0]:
    gwet.delete_cell(c)
gwet.delete_orphan_edges()
gwet.delete_orphan_nodes()
gwet.renumber()
M=gwet.smooth_matrix(K=0.5*np.ones(gwet.Nedges()),dt=1.0)

pos_wet_cells=[gwet.select_cells_nearest(pnt,inside=True) or gwet.select_cells_nearest(pnt,inside=False)
               for pnt in np.c_[seg_tracks.x,seg_tracks.y]]

##

fig=plt.figure(51)
fig.clf()
ax=fig.add_subplot(1,1,1)

binned=np.bincount(pos_wet_cells,minlength=gwet.Ncells())

dens_smooth=binned
for _ in range(20):
    dens_smooth=M.dot(dens_smooth)

ccoll=gwet.plot_cells(values=np.argsort(np.argsort(dens_smooth)),
                      cmap=turbo,ax=ax,lw=0.2)
ccoll.set_edgecolor('face')
ccoll.set_clim([5000,7500])

ax.plot( seg_tracks.x, seg_tracks.y, 'k.',alpha=0.4,ms=1)

ax.axis('equal')
ax.axis('off')
ax.set_position([0,0,1,1])
ax.axis(zoom)
fig.savefig(os.path.join(fig_dir,'spatial-detection-density.png'))

##

seg_tracks['wet_cell']=pos_wet_cells

# how many unique individuals visited each cell?
df1=seg_tracks.groupby(['wet_cell','id'],as_index=False)['id'].first()

cell_to_count=df1.reset_index().groupby('wet_cell').size()
cell_counts=np.zeros( gwet.Ncells(),np.int32)
cell_counts[cell_to_count.index.values] = cell_to_count.values

# occupancy
fig=plt.figure(55)
fig.clf()
ax=fig.add_subplot(1,1,1)

ccoll=gwet.plot_cells(values=cell_counts,cmap=turbo,ax=ax,lw=0.2)
ccoll.set_edgecolor('face')

ax.axis('equal')
ax.axis('off')
ax.set_position([0,0,1,1])
ax.axis(zoom)
plt.colorbar(ccoll,label="# unique individuals")
fig.tight_layout()
fig.savefig(os.path.join(fig_dir,'spatial-occupancy.png'))

##

# Expanded tracks -- 



##

# Distribution of swim_urel
bin_urel=np.bincount(pos_wet_cells,weights=seg_tracks['swim_urel'],minlength=gwet.Ncells())

fig=plt.figure(52)
fig.clf()
ax=fig.add_subplot(1,1,1)

binned=np.bincount(pos_wet_cells,minlength=gwet.Ncells())

dens_smooth=binned+1.0 # pushes areas of no data to 0.0
urel_smooth=bin_urel
for _ in range(60):
    dens_smooth=M.dot(dens_smooth)
    urel_smooth=M.dot(urel_smooth)

bin_mean_urel=urel_smooth/dens_smooth
bin_mean_urel[ np.isnan(bin_mean_urel) ] = 0.0

ccoll=gwet.plot_cells(values=bin_mean_urel,
                      cmap='coolwarm',ax=ax,lw=0.2)
ccoll.set_edgecolor('face')
ccoll.set_clim([-0.15,0.15])
plt.colorbar(ccoll,label='Swim U')

# ax.plot( seg_tracks.x, seg_tracks.y, 'k.',alpha=0.4,ms=1)

ax.axis('equal')
ax.axis('off')
ax.set_position([0,0,1,1])
ax.axis((647119., 647519., 4185681., 4185981.))
fig.savefig(os.path.join(fig_dir,'spatial-swim_urel.png'))

## 

# Distribution of swim_vrel
bin_vrel=np.bincount(pos_wet_cells,weights=seg_tracks['swim_vrel'],minlength=gwet.Ncells())

fig=plt.figure(53)
fig.clf()
ax=fig.add_subplot(1,1,1)

binned=np.bincount(pos_wet_cells,minlength=gwet.Ncells())

dens_smooth=binned+1.0
vrel_smooth=bin_vrel
for _ in range(60):
    dens_smooth=M.dot(dens_smooth)
    vrel_smooth=M.dot(vrel_smooth)

bin_mean_vrel=vrel_smooth/dens_smooth
bin_mean_vrel[ np.isnan(bin_mean_vrel) ] = 0.0

ccoll=gwet.plot_cells(values=bin_mean_vrel,
                      cmap='coolwarm',ax=ax,lw=0.2)
ccoll.set_edgecolor('face')
ccoll.set_clim([-0.15,0.15])
plt.colorbar(ccoll,label='Swim V')

ax.axis('equal')
ax.axis('off')
ax.set_position([0,0,1,1])
ax.axis((647119., 647519., 4185681., 4185981.))
fig.savefig(os.path.join(fig_dir,'spatial-swim_vrel.png'))

## 
