import xarray as xr

##

# nc=xr.open_dataset('ptm2_average.nc_0000.nc')
nc=xr.open_dataset('tmp_untrim/untrim_hydro_Jan2017-t450.nc4',decode_times=False)

## 

# Continuity for a fully wet cell

if 0: # reasonable wet cell for suntans runs
    n=3 # timestep.  Will test the interval with instantaneous outputs at n,n+1
    i=0 # cell
    k=35 # layer. 0-based from bed.
if 0: # untrim wet cell
    n=3 # timestep.  Will test the interval with instantaneous outputs at n,n+1
    i=0 # cell
    k=49 # layer. 0-based from bed.

if 1: # untrim-t450 wet cell
    n=3 # timestep.  Will test the interval with instantaneous outputs at n,n+1
    i=2 # cell
    k=49 # layer. 0-based from bed.

# Change in volume:
vols=nc.Mesh2_face_water_volume.isel(nMesh2_face=i,nMesh2_data_time=[n,n+1],nMesh2_layer_3d=k).values
print(f"[n={n} i={i} k={k}] volume change: {vols[0]} to {vols[1]},  delta={vols[1]-vols[0]}")

# Horizontal fluxes over that time step:
js=nc.Mesh2_face_edges.isel(nMesh2_face=i).values.astype(np.int32)
js=js[js>=0]

## Check that Mesh2_face_edges is 0-based:
face_edges=nc.Mesh2_face_edges.values.astype(np.int32)
# assume that missing value is <0
face_edge_min=face_edges[face_edges>=0].min()
face_edge_max=face_edges[face_edges>=0].max()

if (face_edge_min==0) and (face_edge_max<nc.dims['nMesh2_edge']):
    print("Verified 0-based Mesh2_face_edges")
elif (face_edge_min>0) and (face_edge_max==nc.dims['nMesh2_edge']):
    print("WARNING: Verified 1-based Mesh2_face_edges")
elif (face_edge_min>0) and (face_edge_max<nc.dims['nMesh2_edge']):
    print("WARNING: Inconclusive 0/1-based Mesh2_face_edges")
else:
    print("ERROR: Mesh2_face_edges is not self-consistent, or bad missing value")
    
##

if 1: # untrim partial cell
    n=3 # timestep.  Will test the interval with instantaneous outputs at n,n+1
    i=2 # cell
    k=52 # layer. 0-based from bed.

# Change in volume:
vols=nc.Mesh2_face_water_volume.isel(nMesh2_face=i,nMesh2_data_time=[n,n+1],nMesh2_layer_3d=k).values
print(f"[n={n} i={i} k={k}] volume change: {vols[0]} to {vols[1]},  delta={vols[1]-vols[0]}")

##

# Verify that ctop and water_volume are always consistent, and that
# ctop and cbot are both inclusive, stored as 1-based.

n=3
kti=nc['Mesh2_face_top_layer'].isel(nMesh2_data_time=n).values
kbi=nc['Mesh2_face_bottom_layer'].isel(nMesh2_data_time=n).values
vol=nc['Mesh2_face_water_volume'].isel(nMesh2_data_time=n).values

eta=nc['Mesh2_sea_surface_elevation'].isel(nMesh2_data_time=n).values
depth=nc['Mesh2_face_depth'].values

# vol=np.ma.filled(vol,0.0)

for i in range(len(kti)):
    # kbi inclusive, 1-based
    if kbi[i]<0:
        # dry?
        if kbi[i]!=kti[i]:
            print("WARNING: dry(?) cell has negative kbi, but kti(%d)!=kbi(%d)"%
                  (kti[i],kbi[i]))
        continue
    if np.any( vol[i,:(kbi[i]-1)]!=0.0 ):
        print("WARNING: non-zero volumes below kbi")
    if np.any( vol[i,(kti[i]-1+1):] != 0.0 ):
        print("WARNING: non-zero volumes above kti")
    if np.any( vol[i,kbi[i]-1:kti[i]-1+1] == 0.0 ):
        dz=eta[i] + depth[i] # this comes up nonzero but vol is 0.0?
        # Is this a subgrid thing?
        face_areas=nc['Mesh2_face_wet_area'].isel(nMesh2_face=i,nMesh2_data_time=n).values
        print("WARNING: zero volumes within kbi=%d:kti=%d.  dz=%f"%(kbi[i],kti[i],dz))
        print("         face_areas:",face_areas)
        break


# reading through the untrim netcdf io code:
# face_wet_area is period averaged vertical conveyance area.

##

# Similar but for etop. Slightly more difficult since stagnant water harder to
# discern from empty space.

n=3
ktj=nc['Mesh2_edge_top_layer'].isel(nMesh2_data_time=n).values
kbj=nc['Mesh2_edge_bottom_layer'].isel(nMesh2_data_time=n).values
Q=nc['h_flow_avg'].isel(nMesh2_data_time=n).values

Q=np.ma.filled(Q,0.0)

for j in range(len(ktj)):
    # kbi inclusive, 1-based
    if kbj[j]<0:
        # dry?
        if kbj[j]!=ktj[j]:
            print("WARNING: dry(?) edge has negative kbj, but ktj(%d)!=kbj(%d)"%
                  (ktj[j],kbj[j]))
        continue
    if np.any( Q[j,:(kbj[j]-1)]!=0.0 ):
        print("WARNING: non-zero flows below kbj")
    if np.any( Q[j,(ktj[j]-1+1):] != 0.0 ):
        print("WARNING: non-zero flows above ktj")
    if np.any( Q[j,kbj[j]-1:ktj[j]-1+1] == 0.0 ):
        # This can happen if the water is stagnant, though.
        print("zero flows within kbj:ktj")
