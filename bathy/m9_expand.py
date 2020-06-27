# Look into the feasibility of expanding the bathy
# footprint based on the 9 beams.

from scipy.io import loadmat
import pandas as pd
import glob

import matplotlib.pyplot as plt
from stompy.spatial import proj_utils
from stompy.spatial import proj_utils, wkb2shp, field
import matplotlib.pyplot as plt
import xarray as xr
from stompy.spatial import wkb2shp

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')



## 
# scipy fails to load the mat file.
# m=loadmat('../field/adcp/bathy_data/Bathy_2018.mat',variable_names=[])


#mats=~/src/hor_flow_and_salmon/field/adcp/040518_BT/040518_8BTref/

# So export to ASCII...
# vel: sample #, date/time, frequency, profile type, depth, cell geom and then velocity data.
# snr: sample #, date/time, frequency, profile type, depth, cell geom and then SNR values.

# the sum file is the one that I want.

sum_patts=[ '../field/adcp/bathy_data/Bathy_2018.sum',
            '../field/adcp/040518_BT/040*BT*/*.sum']

summs=[]

def parse_deg(s):
    s=s.replace('\u00b0','').replace('"','').replace("'",'').split()
    f=[float(t) for t in s]
    return f[0] + np.sign(f[0])/60. * (f[1] + f[2]/60.)

for patt in sum_patts:
    for sum_fn in glob.glob(patt):
        print(sum_fn)
        summ=pd.read_csv(sum_fn,
                         parse_dates=['Date/Time'],
                         encoding='latin')

        if summ['Longitude (deg)'].dtype!=np.float64:
            summ['Latitude (deg)']=summ['Latitude (deg)'].apply(parse_deg)
            summ['Longitude (deg)']=summ['Longitude (deg)'].apply(parse_deg)
        summs.append(summ)
        

summ=pd.concat(summs)

##         
# In some cases the lat/lon are not only in DMS, but with a latin-encoded
# degree symbol.


# to UTC, with the assumption that ADCP recorded in PDT.
# consistent with calibration script.
summ['time']=summ['Date/Time'] + np.timedelta64(7,'h')

##

ll=summ.loc[:,['Longitude (deg)','Latitude (deg)']]

xy=proj_utils.mapper('WGS84','EPSG:26910')(ll)

summ['x']=xy[:,0]
summ['y']=xy[:,1]

invalid=ll.values[:,0]==0.0

summ=summ[~invalid].copy()

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ax.plot(summ.x, summ.y, 'r.')

##

# This is the value used in creating the summary outputs
xducer_depth=0.41


##
summ_sel=summ
#summ_sel=summ.iloc[4900:5080,:]
summ_sel=summ_sel.copy()

# Subset

summ_sel['theta_janus']=25 * np.pi/180.

vb=summ_sel.copy()
vb['z']=-summ_sel['VB Depth (m)']
vb['src']='vb'

# Could try to get fancier and include pitch and roll.
# Pitch has a mode around +5 deg.
# Roll has a mode around +3 deg.
# With beam 1 at 0 degree offset,
# and pitch nearly constant, and larger than roll, can
# get a very rough sense by modifying the janus angle.

# geographic to mathematic convention
# Tested +-15 of declination, but no improvement
phi=( 90-summ_sel['Heading (deg)']) * np.pi/180.

# Beam 1 at 1MHz is best at 0 degrees.
# And at 3MHz, 45 degrees.

# Including pitch and roll did not improve results in a measurable
# way.

b1=summ_sel.copy()
b1['phi']=phi + 0.0 + (b1['Frequency (MHz)']=='3MHz') * (45 * np.pi/180.)
b1['z']=-summ_sel['BT Beam1 Depth (m)']
b1['src']='bt1'

b2=summ_sel.copy()
b2['z']=-summ_sel['BT Beam2 Depth (m)']
b2['phi']=b1['phi']+np.pi/2
b2['src']='bt2'

b3=summ_sel.copy()
b3['phi']=b1['phi']+np.pi
b3['z']=-summ_sel['BT Beam3 Depth (m)']
b3['src']='bt3'

b4=summ_sel.copy()
b4['phi']=b1['phi']+3*np.pi/2
b4['z']=-summ_sel['BT Beam4 Depth (m)']
b4['src']='bt4'

all_b=pd.concat( [b1,b2,b3,b4] )


# Assume that the reported depth already accounts for the janus angle,
# This is then the radial offset from the GPS fix to the janus beam
all_b['depth_below_transducer']=(-all_b['z'])-xducer_depth
all_b['radius']=np.tan(all_b['theta_janus']) * all_b['depth_below_transducer']
all_b['x'] += all_b.radius * np.cos(all_b.phi)
all_b['y'] += all_b.radius * np.sin(all_b.phi)

# b1['x'] += rad_janus * np.cos(b1_angle + phi)
# b1['y'] += rad_janus * np.sin(b1_angle + phi)
# b2['x'] += rad_janus * np.cos(b2_angle + phi)
# b2['y'] += rad_janus * np.sin(b2_angle + phi)
# 
# b3['x'] += rad_janus * np.cos(b3_angle + phi)
# b3['y'] += rad_janus * np.sin(b3_angle + phi)
# 
# b4['x'] += rad_janus * np.cos(b4_angle + phi)
# b4['y'] += rad_janus * np.sin(b4_angle + phi)

# For tuning, limit to vb, a single side beam, and only
# 1 frequency.

#all_beams=pd.concat( [vb,b1,b2,b3,b4] )

all_beams=pd.concat( [vb, all_b] )


## Now adjust for water level.
# Assume

utils.path("../model/suntans")

import common

t_start=all_beams['time'].values.min()
t_end  =all_beams['time'].values.max()

oh1_stage=common.oh1_stage(t_start,t_end)
#sjd_stage=common.sjd_stage(t_start,t_end)# doesn't exist..
msd_stage=common.msd_stage(t_start,t_end)

##

stage=np.interp( utils.to_dnum(all_beams.time.values),
                 utils.to_dnum(oh1_stage.time.values), oh1_stage.stage_m.values )

all_beams['z_adj']=all_beams['z']+stage

## 

# There's a good bit of drop here, ~ 50cm
# OH1 is much closer, though from the model run the biggest
# drop is at the barrier
# For current purposes of fixing the hole.

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

scat=ax.scatter(all_beams.x, all_beams.y, 60, all_beams.z_adj, cmap=turbo)
scat.set_clim([-6,0])

ax.axis('equal')
ax.axis((647252., 647288, 4185938, 4185973))
#ax.axis((647236., 647272, 4185916, 4185952))

##

condensed=all_beams.loc[:, ['Date/Time','Frequency (MHz)', 'Heading (deg)', 'Latitude (deg)',
                            'Longitude (deg)','Pitch (deg)', 'Profile Type','Roll (deg)', 'Sample #',
                            'depth_below_transducer', 'phi','radius', 'src', 'theta_janus', 'time',
                            'x', 'y', 'z', 'z_adj'] ]
condensed.to_csv('m9-expand-all_beams.csv',index=False)

##

# How does this compare wih the grid itself?

g=unstructured_grid.UnstructuredGrid.read_ugrid('../model/suntans/snubby-08-edit60-with_bathysmooth.nc')
## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

scat=ax.scatter(all_beams.x, all_beams.y, 60, all_beams.z_adj, cmap=turbo)
clim=[-6,0]

scat.set_clim(clim)

coll=g.plot_cells(values='z_bed',clim=clim,cmap=turbo,zorder=-2)

ax.axis('equal')
ax.axis((647252., 647288, 4185938, 4185973))
#ax.axis((647236., 647272, 4185916, 4185952))

##

# just use the computational grid
g=unstructured_grid.UnstructuredGrid.read_ugrid('../model/grid/snubby_junction/snubby-08-edit60.nc')
    
##

bcs=wkb2shp.shp2geom("../model/grid/snubby_junction/forcing-snubby-01.shp")

##

dx=1.0
dy=1.0

xmin,xmax,ymin,ymax=g.bounds()

xmin= dx*np.floor(xmin/dx)
xmax= dx*np.ceil(xmax/dx)

ymin= dy*np.floor(ymin/dy)
ymax= dy*np.ceil(ymax/dy)

# Use the computational grid to set boundaries, but run the smoothing on the
# DEM so we can mix it in after the fact
gr=unstructured_grid.UnstructuredGrid(max_sides=4)

# Way overkill..
gr.add_rectilinear( p0=[xmin,ymin], p1=[xmax,ymax],
                    nx=int((xmax-xmin)/dx+1),
                    ny=int((ymax-ymin)/dy+1) )

##
bounds=g.boundary_polygon()

sel=gr.select_cells_intersecting(bounds)

##
for c in np.nonzero(~sel)[0]:
    gr.delete_cell(c)

gr.delete_orphan_edges()
gr.delete_orphan_nodes()
gr.renumber()

##

# Can get better interpolation on the dual
gd=gr.create_dual(create_cells=True)

##

zoom=(647088.821657404, 647702.7390601198, 4185484.2206095303, 4185941.6880934904)

plt.figure(10).clf()

fig,ax=plt.subplots(1,1,num=10)

gr.plot_edges( ax=ax,lw=0.3,color='k')

## 
# Three BCs, six total sets of edges.

edge_sets=[]
for bc in bcs:
    edge_sets.append( gr.select_edges_by_polyline(bc['geom']) )

OR_left =np.array(bcs['geom'][1].coords)[0] # left end at OR
OR_right=np.array(bcs['geom'][1].coords)[-1] # right end at OR
SJ_up_left=np.array(bcs['geom'][0].coords)[0]
SJ_up_right=np.array(bcs['geom'][0].coords)[-1]
SJ_dn_right=np.array(bcs['geom'][2].coords)[-1]
SJ_dn_left=np.array(bcs['geom'][2].coords)[0]

river_left = gr.select_nodes_boundary_segment( coords=[OR_left,SJ_up_left] ) 
river_right= gr.select_nodes_boundary_segment( coords=[SJ_up_right,SJ_dn_right] )
river_split= gr.select_nodes_boundary_segment( coords=[SJ_dn_left,OR_right] )

sj_up  = gr.select_nodes_boundary_segment( coords=[SJ_up_left,SJ_up_right] ) 
or_down= gr.select_nodes_boundary_segment( coords=[OR_right, OR_left] )
sj_down= gr.select_nodes_boundary_segment( coords=[SJ_dn_right,SJ_dn_left] )

def to_cells(nodes):
    return np.unique( [ c for n in nodes for c in gr.node_to_cells(n) ] )

# cells:
river_left_cells =to_cells(river_left)
river_right_cells=to_cells(river_right)
river_split_cells=to_cells(river_split)

sj_up_cells =to_cells(sj_up)
or_down_cells=to_cells(or_down)
sj_down_cells=to_cells(sj_down)

##
plt.figure(12).clf()
gr.plot_edges(lw=0.5,color='k')
gr.plot_cells(mask=sj_up_cells,color='r',alpha=0.5)
## 

from stompy.model import unstructured_diffuser

diff=unstructured_diffuser.Diffuser(gr)

for c in river_left_cells:
    diff.set_dirichlet(0,cell=c)

for c in river_right_cells:
    diff.set_dirichlet(100,cell=c)

for c in river_split_cells:
    diff.set_dirichlet(50,cell=c)
    
diff.construct_linear_system()
diff.solve_linear_system(animate=False)

psi=diff.C_solved

##

# Try for the velocity potential, too.

diff2=unstructured_diffuser.Diffuser(gr)

for c in sj_up_cells:
    diff2.set_dirichlet(0,cell=c)


# Haven't worked through the math and geometry
# to know the correct way to relate Q choices
# and potential choices.
# having them equal looks pretty good
# 110/100 is not good, and 90/100 is not good.
# Trial and error to get cells near the junction
# to look orthogonal, leads to -97 and -100.
for c in sj_down_cells:
    diff2.set_dirichlet(-97,cell=c)

for c in or_down_cells:
    diff2.set_dirichlet(-100,cell=c)
    
diff2.construct_linear_system()
diff2.solve_linear_system(animate=False)

phi=diff2.C_solved

## 

psi_node=psi[ gd.nodes['dual_cell'] ]
phi_node=phi[ gd.nodes['dual_cell'] ]

##

plt.figure(10).clf()
fig,ax=plt.subplots(1,1,num=10)

img1=dem_crop.plot(clim=[-5,2.2],ax=ax,cmap='gray')

gd.contour_node_values(psi_node,40,linewidths=0.5,colors='orange')
gd.contour_node_values(phi_node,300,linewidths=0.5,colors='red')

ax.axis(zoom)

##

# Convert that to rasters
phi_fld=field.XYZField(gr.cells_center(),phi).rectify()
psi_fld=field.XYZField(gr.cells_center(),psi).rectify()

##

# add psi and phi coordinates to condensed.
xy=condensed.loc[:,['x','y']].values

phi_fld.default_interpolation='linear'
psi_fld.default_interpolation='linear'
condensed['phi']=phi_fld(xy)
condensed['psi']=psi_fld(xy)

##

plt.figure(12).clf()
fig,ax=plt.subplots(num=12)

ax.scatter(condensed['phi'],
           condensed['psi'],
           30,
           condensed['z_adj'],
           cmap=turbo)

##

plt.figure(1).clf()
## 

img=z_fld.plot(cmap=turbo)

valid=np.isfinite( condensed['phi'].values * condensed['psi'].values )

psi_scale=3
gridded=griddata( np.c_[ condensed['phi'].values,
                         psi_scale*condensed['psi'].values][valid],
                  condensed['z_adj'][valid],
                  np.c_[phi,psi_scale*psi] )

z_fld=field.XYZField(gr.cells_center(),gridded).rectify()

z_fld.smooth_by_convolution(kernel_size=3,iterations=1)

img.set_array(z_fld.F)

plt.draw()

## 

z_fld.write_gdal('adcp-m9-expand-1m.tif')

