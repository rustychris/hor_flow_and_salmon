"""
Plot tracks after obvious bad tracks removed.
"""
import os
import glob
from scipy.signal import medfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import track_common
from scipy.interpolate import interp1d

from stompy import utils
from scipy.stats import gaussian_kde
from matplotlib import gridspec, collections, widgets
from stompy.spatial import field
import seaborn as sns
import stompy.plot.cmap as scmap
from stompy.plot import plot_utils
turbo=scmap.load_gradient('turbo.cpt')


utils.path("../../manuscripts/swimming/fig_common/")

##

input_path="screen_final"

df=track_common.read_from_folder(input_path)

vel_suffix='_top2m'

df['track'].apply(track_common.calc_velocities,
                  model_u='model_u'+vel_suffix,model_v='model_v'+vel_suffix)
track_common.clip_to_analysis_polygon(df,'track')

##
# matches folder in track_analyses
fig_date="2020917"

# For plots that are not tied to a specific velocity (see end of this file)
fig_dir_gen="fig_analysis-"+fig_date
fig_dir=os.path.join(fig_dir_gen, vel_suffix[1:])
for d in [fig_dir_gen,fig_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

                
##     PLOTS    ##

hydros=pd.read_csv('yap-positions.csv')

## 
dem=field.GdalGrid("../../bathy/junction-composite-20200604-w_smooth.tif")

##

# Concatenate into a single dataframe for some of these plots
for idx,row in df.iterrows():
    row['track']['id'] = idx
    
seg_tracks=pd.concat( [ track.iloc[:-1,:]
                        for track in df['track'].values ] )

##

# Plot the global distribution of position standard deviations

# This isn't actually dependent on hydro velocity, but
# I'm sticking it here to keep it in sync with any choice of
# sample selection (like trim_to_analysis_period)
all_locs=pd.concat( df['track'].values )
all_locs['sd']=np.sqrt( all_locs.x_sd**2 + all_locs.y_sd**2)

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

bins = np.logspace(-2,3,100)
ax.set_xscale('log')
ax.hist(all_locs['sd'],bins=bins)

ax.set_xlabel('Standard deviation of position estimates')
ax.set_ylabel('Count of position estimates')

q1,q2,q3=np.percentile( all_locs.sd, [25,50,75] )

l25=ax.axvline(q1,ymin=0.15,color='k',lw=0.5)
l50=ax.axvline(q2,ymin=0.15,color='k',lw=0.5)
l75=ax.axvline(q3,ymin=0.15,color='k',lw=0.5)

# plot_utils.annotate_line(l25,'25%',norm_position=0.6)
ax.text( q1,0.05, "25%", transform=l25.get_transform(),rotation=90,ha='center')
ax.text( q2,0.05, "50%", transform=l50.get_transform(),rotation=90,ha='center')
ax.text( q3,0.05, "75%", transform=l75.get_transform(),rotation=90,ha='center')

print("Quartiles of std. dev. of position estimates: [%.2f %.2f %.2f]"%(q1,q2,q3))

#fig.savefig(os.path.join(fig_dir,'all_locs-stddev.png'))

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


# Global relationship between swim speed and hydro speed
def figure_swim_hydro_speed(num=23,
                            bandwith=0.25,
                            seg_tracks=seg_tracks,
                            weights=None,
                            label=""):
    plt.figure(num).clf()
    fig,axs=plt.subplots(1,3,num=num)
    #ax_mag,ax_u,ax_v,ax_usign=axs
    ax_mag,ax_v,ax_usign=axs # ax_u isn't that useful
    
    fig.set_size_inches((8,2.75),forward=True)

    swim_speed =np.sqrt( seg_tracks['swim_u'].values**2 + seg_tracks['swim_v'].values**2)
    hydro_speed=np.sqrt( seg_tracks['model_u'+vel_suffix].values**2 +
                         seg_tracks['model_v'+vel_suffix].values**2)

    for ax,dep in [ (ax_mag,swim_speed),
                    # (ax_u, np.abs(seg_tracks['swim_urel'].values)),
                    (ax_v, np.abs(seg_tracks['swim_vrel'].values)),
                    (ax_usign,seg_tracks['swim_urel'].values)]:
        ax.plot( hydro_speed, dep, '.', color='tab:blue',ms=1,alpha=0.2, zorder=-1)
        order=np.argsort(hydro_speed)
        N=201

        if weights is not None:
            hyd_med=track_common.medfilt_weighted( hydro_speed[order], N,weights=weights)
            dep_med=track_common.medfilt_weighted( dep[order], N, weights=weights)
        else:
            hyd_med=medfilt( hydro_speed[order], N)
            dep_med=medfilt( dep[order], N)
            
        ax.plot( hyd_med[N:-N], dep_med[N:-N], 'k-',zorder=1)
            
        ax.set_xlabel('Water speed (m/s)')
        ax.axis(xmin=0,xmax=0.8)
        if ax!=ax_usign:
            ax.axis(ymin=0,ymax=0.5)
            ax.set_yticks([0,0.25,0.5])
        else:
            ax.axis(xmin=0,xmax=0.8,ymin=-0.5,ymax=0.5)
            ax.set_yticks([-0.5,-0.25,0,0.25,0.5])
            ax.axhline(0,color='k',lw=1.2, zorder=0)

    for ax,panel in zip(axs,"abcdef"):
        ax.text(0.03,0.9,f"({panel})",transform=ax.transAxes,
                fontweight='bold')
            
    ax_mag.set_ylabel('m s$^{-1}$')
        
    ax_mag.set_title('Swim speed')
    # ax_u.set_title(r'$|lon|$')
    ax_v.set_title('Lateral Speed')
    ax_usign.set_title('Longitudinal Velocity')
    
    # plt.setp(ax_u.get_yticklabels(), visible=0)
    plt.setp(ax_v.get_yticklabels(), visible=0)
    ax_usign.yaxis.tick_right()
    ax_usign.yaxis.set_label_position("right")
    ax_usign.set_ylabel('m s$^{-1}$')

    fig.subplots_adjust(left=0.095,right=0.90,top=0.88,bottom=0.18)
    return fig

fig=figure_swim_hydro_speed()
# all_segments-swim_hydro_speed.png was when it had 4 panels.
fig.savefig(os.path.join(fig_dir,"all_segments-swim_hydro_speed-3panels.png"),dpi=200)

##

seg_counts=seg_tracks.groupby(seg_tracks.id).size()
weights=1./seg_counts[ seg_tracks.id]

fig=figure_swim_hydro_speed(weights=weights,num=24)
fig.savefig(os.path.join(fig_dir,"all_segments-swim_hydro_speed-per_tag.png"),dpi=200)

##
model_u='model_u'+vel_suffix
model_v='model_v'+vel_suffix

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
plots=fig_track_swimming(df.loc['7A96','track'],
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

# All tracks, but highlight a few with indication of swimming
num=70
plt.figure(num).clf()
fig,ax=plt.subplots(num=num)
fig.set_size_inches([6.5,5.5],forward=True)

# All segments:
segs=[]
for track in df['track']:
    segs.append( track.loc[:, ['x','y']].values )
ax.add_collection( collections.LineCollection(segs,lw=0.5,alpha=0.3,color='k') )    
ax.axis('equal')


import fig_common
img,cset=fig_common.composite_aerial_and_contours(ax)

ax.axis( (647156., 647431., 4185688., 4185919))

ax.axis('off')
ax.set_position([0,0,1,1])

qsets=[]

highlights=[ ['7A96','k','Lateral'],
             ['7ACD','g','Passive'],
             ['75BA','r','Pos. Rheotaxis'],
             ['76AF','b','Neg. Rheotaxis'] ]

# Highlight a few tracks with swim quiver
for tag, color, label in highlights:
    track=df.loc[tag,'track']
    xyuv=track.loc[:, ['x','y','swim_u','swim_v']].values

    ax.plot(xyuv[:,0], xyuv[:,1], color=color,lw=2.,alpha=0.6,zorder=1)

    d=utils.dist_along( xyuv[:,:2] )
    dx=20.0 # put an arrow every dx meters along track.
    int_xyuv=interp1d(d, xyuv, axis=0)( np.arange(d.min(),d.max(),dx) )
    qset=ax.quiver( int_xyuv[:,0], int_xyuv[:,1],
                    int_xyuv[:,2], int_xyuv[:,3],
                    angles='xy',width=0.007,color=color,
                    headaxislength=2.5, headlength=2.5,headwidth=2.5,
                    # scale_units='width', scale=7,
                    scale_units='xy',scale=0.025,
                    zorder=2)
    qsets.append(qset)

# quiverkey seems more reliable in geographic coordinates
x0=647180
y0=4185759
dy=11
for i,(tag,color,label) in enumerate(highlights):
    ax.quiverkey(qsets[i],x0,y0-dy*i, 0.3,label,coordinates='data',
                 labelpos='E')

ax.text(x0-16,y0+0.8*dy,"0.3 m s$^{-1}$")

fig.savefig(os.path.join( fig_dir,"track-behavior-examples.png"),dpi=200)

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

##  FIGURES across ALL VELOCITY SUFFIXES  

# Histogram / Density plot for mean swim speed, showing all 3
# choices of velocity in one plot:
segs_all_vel=[]
suffixes=['_top1m','_top2m','_davg']
nice_names={ '_top1m':'Top 1m',
             '_top2m':'Top 2m',
             '_davg' :'Depth\naverage'}

for vel_suffix in suffixes:
    df_tmp=track_common.read_from_folder(input_path)
    df_tmp['track'].apply(track_common.calc_velocities,
                          model_u='model_u'+vel_suffix,model_v='model_v'+vel_suffix)
    track_common.clip_to_analysis_polygon(df_tmp,'track')

    # Concatenate into a single dataframe for some of these plots
    for idx,row in df_tmp.iterrows():
        row['track']['id'] = idx
    
    segs_tmp=pd.concat( [ track.iloc[:-1,:]
                          for track in df_tmp['track'].values ] )
    segs_tmp['vel']=vel_suffix
    segs_all_vel.append(segs_tmp)

seg_all_vel=pd.concat(segs_all_vel)
seg_all_vel['swim_speed']=np.sqrt( seg_all_vel['swim_u']**2 + seg_all_vel['swim_v']**2 )
##

plt.figure(60).clf()
fig,ax=plt.subplots(num=60)
fig.set_size_inches((4,2.75),forward=True)

umax=0.9
N=401 # number of bins on non-negative side
sym_bins=np.linspace(-umax,umax,2*N-1) # symmetric to simplify mirroring

from scipy.stats import gaussian_kde

for suffix in suffixes[::-1]:
    sel=seg_all_vel['vel']==suffix
    speeds=seg_all_vel.loc[sel,'swim_speed']
    
    kde=gaussian_kde(speeds.values,bw_method='scott')
    Zsym=kde(sym_bins)

    # Mirror gaussian at origin:
    bins=sym_bins[N-1:]
    Z=Zsym[N-1:].copy()
    Z[:] += Zsym[N-1::-1]
    ax.plot(bins,Z,label=nice_names[suffix])

ax.legend(frameon=False)
ax.axis(xmin=0,xmax=umax,ymin=0)
ax.spines['top'].set_visible(0)
ax.spines['right'].set_visible(0)
ax.set_xlabel('Swim speed (m s$^{-1}$)')
ax.set_ylabel('Probability density')
fig.subplots_adjust(left=0.14,bottom=0.18,top=0.95,right=0.95)

fig.savefig(os.path.join(fig_dir_gen,"dist_swim_speed-3-verticals.png"),
            dpi=200)
##

# Similar density plot for rheotaxis:

plt.figure(61).clf()
fig,ax=plt.subplots(num=61)
fig.set_size_inches((4,2.75),forward=True)

umax=0.9
N=401 # number of bins on non-negative side
sym_bins=np.linspace(-umax,umax,2*N-1) # symmetric to simplify mirroring

from scipy.stats import gaussian_kde

seg_counts=seg_all_vel.groupby(seg_all_vel.id).size()
weights=1./seg_counts[seg_all_vel.id]
seg_all_vel['weight_tag']=weights.values

def boot_median(data_weight,reps=1000):
    """
    Bootstrapping confidence intervals for the median
    with weighted data points
    """
    boot_medians=[]
    # to numpy land for speed
    N=len(data_weight)
    for rep in range(1000):
        sample_idxs=np.random.randint(0,N,N)
        sample=data_weight[sample_idxs,:]
        order=np.argsort(sample[:,0])
        sample=sample[order,:]
        cumsum = sample[:,1].cumsum()
        cutoff = cumsum[-1]/2.0
        median = sample[cumsum >= cutoff,0][0]
        boot_medians.append(median)
    median=np.mean(boot_medians)
    cis=np.percentile(boot_medians,[5,95])
    return median,cis
    
for suffix in suffixes[::-1]:
    sel=seg_all_vel['vel']==suffix
    speeds=seg_all_vel.loc[sel,'swim_urel']
    weights=seg_all_vel.loc[sel,'weight_tag']

    kde=gaussian_kde(speeds.values,bw_method='scott',
                     weights=weights)
    Zsym=kde(sym_bins)

    ax.plot(sym_bins,Zsym,label=nice_names[suffix])

    pos_rheo_pct=100*weights[speeds<0].sum()/weights.sum()
    neg_rheo_pct=100*weights[speeds>0].sum()/weights.sum()
    print(f"Velocity {suffix}: pos rheo: {pos_rheo_pct:.1f}%, neg rheo {neg_rheo_pct:.1f}%")

    median,cis = boot_median( seg_all_vel.loc[sel,['swim_urel','weight_tag']].values,
                              1000)
    print(f"  Median swim_urel: {median:.3f} [{cis[0]:.3f}, {cis[1]:.3f}] m/s")
    

ax.legend(frameon=False)
ax.axis(xmin=-umax,xmax=umax,ymin=0)
ax.spines['top'].set_visible(0)
ax.spines['right'].set_visible(0)
ax.set_xlabel('Downstream swim velocity (m s$^{-1}$)')
ax.set_ylabel('Probability density')
ax.axvline(0,color='0.4',zorder=-1,lw=1.0)
fig.subplots_adjust(left=0.14,bottom=0.18,top=0.95,right=0.95)

fig.savefig(os.path.join(fig_dir_gen,"dist_urel-3-verticals.png"),
            dpi=200)

##

plt.figure(62).clf()
fig,ax=plt.subplots(num=62)
fig.set_size_inches((4,2.75),forward=True)

umax=0.9
N=401 # number of bins on non-negative side
sym_bins=np.linspace(-umax,umax,2*N-1) # symmetric to simplify mirroring

from scipy.stats import gaussian_kde

seg_counts=seg_all_vel.groupby(seg_all_vel.id).size()
weights=1./seg_counts[seg_all_vel.id]
seg_all_vel['weight_tag']=weights.values

for suffix in suffixes[::-1]:
    sel=seg_all_vel['vel']==suffix
    speeds=seg_all_vel.loc[sel,'swim_vrel']
    weights=seg_all_vel.loc[sel,'weight_tag']

    kde=gaussian_kde(speeds.values,bw_method='scott',
                     weights=weights)
    Zsym=kde(sym_bins)

    ax.plot(sym_bins,Zsym,label=nice_names[suffix])

    median,cis = boot_median( seg_all_vel.loc[sel,['swim_vrel','weight_tag']].values,
                              1000)
    print(f"Velocity {suffix}: left swimming: {pos_rheo_pct:.1f}%, neg rheo {neg_rheo_pct:.1f}%")
    print(f"  Median swim_vrel: {median:.3f} [{cis[0]:.3f}, {cis[1]:.3f}] m/s")
    

ax.legend(frameon=False)
ax.axis(xmin=-umax,xmax=umax,ymin=0)
ax.spines['top'].set_visible(0)
ax.spines['right'].set_visible(0)
ax.set_xlabel('Lateral swim velocity (m s$^{-1}$)')
ax.set_ylabel('Probability density')
ax.axvline(0,color='0.4',zorder=-1,lw=1.0)
fig.subplots_adjust(left=0.14,bottom=0.18,top=0.95,right=0.95)

fig.savefig(os.path.join(fig_dir_gen,"dist_vrel-3-verticals.png"),
            dpi=200)

##

# Toy calculation of what would happen if position estimates were
# iid.

# Say I have two position estimates, 5 s apart, each with a
# S.D. of 1.4m.
# Take that to be sdx=sdy=1.
N=10000

errAx=np.random.normal(loc=0,scale=1.0,size=N)
errAy=np.random.normal(loc=0,scale=1.0,size=N)

errBx=np.random.normal(loc=0,scale=1.0,size=N)
errBy=np.random.normal(loc=0,scale=1.0,size=N)

dt=5.0

vel=np.sqrt( (errBx - errAx)**2 + (errBy - errAy)**2 ) / dt

print(f"{vel.mean():.3f} m/s +- {vel.std():.3f}")

##

# How does swim speed change when calculated over longer step sizes?

df_tmp=track_common.read_from_folder(input_path)
track_common.clip_to_analysis_polygon(df_tmp,'track')

for vel_suffix in ['_top1m','_top2m','_davg']:
    seg_multistrides=[]

    for idx,row in utils.progress(df_tmp.iterrows()):
        track=row['track']
        track['id'] = idx

        for stride in range(1,5):
            for offset in range(stride):
                sub_track=track.iloc[offset::stride].copy()
                track_common.calc_velocities(sub_track,
                                             model_u='model_u'+vel_suffix,model_v='model_v'+vel_suffix)
                seg_multistrides.append( sub_track.iloc[:-1,:].loc[:,['vel_dt','ground_u','ground_v','swim_u','swim_v',
                                                                      'swim_urel','swim_vrel']] )
    segs_ms=pd.concat(seg_multistrides)

    segs_ms['swim_speed']=np.sqrt( segs_ms['swim_u']**2 + segs_ms['swim_v']**2 )

    segs_ms['n_pings']=np.round(segs_ms['vel_dt']/5.0)
    sel=segs_ms['n_pings']<=10

    plt.figure(1).clf()

    sns.boxplot(x='n_pings',y='swim_speed',data=segs_ms.loc[sel,:])

    speed_medians=segs_ms.loc[sel,:].groupby('n_pings')['swim_speed'].median()
    print(f"Median swim speed, unweighted, velocity suffix f{vel_suffix}")
    print(speed_medians)

    # 0.216 m/s for 5 s interval
    # 0.180 m/s for 50 s interval.

