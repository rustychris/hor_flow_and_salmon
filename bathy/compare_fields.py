from stompy.spatial import field
import stompy.plot.cmap as scmap
import bathy
dem=bathy.dem()
cmap=scmap.load_gradient('hot_desaturated.cpt')

##
clip=(646966, 647602, 4185504, 4186080)

tile=dem.extract_tile(clip)
##

txy00=dict(name='ADCP RBF cluster=3k med=3',f=field.GdalGrid('adcp-rbf-cluster-med3.tif'))
txy01=dict(name='DEM RBF qtracks',f=field.GdalGrid('tile-rbf-qtracks.tif'))
txy02=dict(name='DEM RBF cluster',f=field.GdalGrid('tile-rbf-cluster.tif'))
txy03=dict(name='DEM RBF cluster, 3x3 median',f=field.GdalGrid('tile-rbf-cluster-med3.tif'))
zha00=dict(name='Zhang',f=field.GdalGrid('zhang-dem.tif'))
zbd00=dict(name='Zhang bidir',f=field.GdalGrid('zhang-dem-bidir.tif'))

candidates=[txy00,txy01,txy02,txy03,zha00,zbd00]
truth=tile

##

# bitmask of samples which are valid in all outputs
mask=field.SimpleGrid(extents=truth.extents, F=np.isfinite(truth.F))

# Get them all on the same grid, interpolating onto the tile
for cand in candidates:
    f=cand['f']

    if (f.extents==truth.extents) and (f.dx==truth.dx) and (f.dy==truth.dy):
        cand['fm']=f
    else:
        print("%s not aligned.  Interpolating"%cand['name'])
        cand['fm']=f.extract_tile(match=truth,interpolation='bilinear')

    mask.F=mask.F & np.isfinite(cand['fm'].F)

##
fig_dir="figs20180718"
os.path.exists(fig_dir) or os.mkdir(fig_dir)

for i,cand in enumerate(candidates):
    fig=plt.figure(20+i)
    fig.clf()
    fig.set_size_inches([12,5],forward=True)

    ax=fig.add_subplot(1,3,1)
    F=cand['fm'].F - truth.F
    F[~mask.F]=np.nan # only plot the parts we're comparing
    F=np.ma.masked_invalid(F)
    delta=field.SimpleGrid(extents=truth.extents,F=F)
    delta.plot(ax=ax,cmap='seismic',vmin=-1,vmax=1)
    ax.set_title(cand['name'])
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)

    ax_dem=fig.add_subplot(1,3,2)
    cand['fm'].plot(ax=ax_dem,cmap='jet',vmin=-8,vmax=3)
    cand['fm'].plot_hillshade(ax=ax_dem,z_factor=3)
    ax_dem.xaxis.set_visible(0)
    ax_dem.yaxis.set_visible(0)

    errors=F[np.isfinite(F)]
    ax_h=fig.add_subplot(2,3,3)
    ax_h.hist(errors,bins=np.linspace(-1,1,150) )

    ax_t=fig.add_subplot(2,3,6)
    lines=[]
    lines.append("RMSE: %.2f m"%utils.rms(errors))
    ax_t.text(0.02,0.9,"\n".join(lines),transform=ax_t.transAxes)
    plt.setp(ax_t.spines.values(),visible=0)
    ax_t.xaxis.set_visible(0)
    ax_t.yaxis.set_visible(0)
    fig.tight_layout()

    fig.savefig(os.path.join(fig_dir,'scores-%s.png'%cand['name']))
