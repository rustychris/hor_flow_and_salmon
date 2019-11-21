from matplotlib import collections
from stompy.spatial import field

##

dem=field.GdalGrid("../../../bathy/junction-composite-20190117-no_adcp.tif")

##
# this still has many crazy tracks.
cleaned=pd.read_csv('cleaned_half_meter.csv')
grp=cleaned.groupby('ID')

# this is mostly cleaned up.
segs=pd.read_csv('segments_2m-model20190314.csv')

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

for fish,grp in cleaned.groupby('ID'):
    plt.plot(grp['X'],grp['Y'],lw=0.2,alpha=0.5,color='r',
             zorder=1)

for fish,grp in segs.groupby('id'):
    xyxy = np.array( [ [grp['x1'],grp['y1']], [grp['x2'],grp['y2']] ] ).transpose(2,0,1)
    seg_coll=collections.LineCollection( xyxy, zorder=2,lw=1.5,color='b' )
    ax.add_collection(seg_coll)
    
ax.axis('equal')

dc=dem.crop(ax.axis())
img=dc.plot(cmap='gray')
img2=dc.plot_hillshade(z_factor=3)

fig.savefig('raw_vs_filtered.png')

##

# For Mike, compile two dataframes:
#   
