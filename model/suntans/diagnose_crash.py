import numpy as np
import matplotlib.pyplot as plt
from stompy.grid import unstructured_grid
import xarray as xr
##
proc=0
j=13333
c=4129

ds=xr.open_dataset('runs/steady_008dt1/Estuary_SUNTANS.nc.nc.%d'%proc)

g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)
##
umag=np.sqrt(ds.uc.isel(time=-1)**2 + ds.vc.isel(time=-1)**2)
umag_max=umag.max(dim='Nk')

if j is not None:
    center=g.edges_center()[j]
    scale=g.edges_length()[j]
elif c is not None:
    center=g.cells_center()[c]
    scale=2*np.sqrt(g.cells_area())[c]
pad=30*scale
pad_small=pad/5
zoom=(center[0]-pad,center[0]+pad, center[1]-pad, center[1]+pad)
zoom_in=(center[0]-pad_small,center[0]+pad_small, center[1]-pad_small, center[1]+pad_small)



# where is the bad cell?
plt.figure(1).clf()
fig,(ax,ax_d)=plt.subplots(1,2,num=1,sharex=True,sharey=True)

g.plot_edges(ax=ax,color='k',lw=0.4,clip=zoom)
g.plot_edges(ax=ax,labeler='id',clip=zoom_in)

ucoll=g.plot_cells(ax=ax,values=umag_max,clip=zoom,
                   labeler=lambda i,r: str(i))

# d_vals=ds.dv.values
d_vals=ds.eta.isel(time=t).values
dcoll=g.plot_cells(ax=ax_d,values=d_vals,clip=zoom_in,
                   labeler=lambda i,r: "%.3f"%d_vals[i])
plt.setp(ax_d.texts,ha='center',clip_on=True)
plt.setp(ax.texts,ha='center',clip_on=True)


ax.plot( [center[0]], [center[1]], 'mo',ms=4)

ax.axis('equal')
ax.axis(zoom_in)

##


# ktop_wet=19.  50072 high velocity in 19, 20
ec=g.edges_center()
k=19
Uslice=ds.U.isel(time=-10).values[k,:]
invalid=np.isnan(Uslice)
Uslice[invalid]=0.0
eU=ds.n1.values * Uslice
eV=ds.n2.values * Uslice

emask=g.edge_clip_mask(zoom_in)

del ax_d.collections[1:]

qset=ax_d.quiver( ec[emask,0],ec[emask,1],eU[emask],eV[emask])
ax_d.quiverkey(qset,0.1,0.95,1.0,"1 m/s",coordinates='axes')

ax_d.set_title('k=%d'%k)

cmask=g.cell_clip_mask(zoom_in)
cc=g.cells_center()
cU=ds.uc.isel(time=t).values[k,:]
cV=ds.vc.isel(time=t).values[k,:]
qset2=ax_d.quiver( cc[cmask,0],cc[cmask,1],cU[cmask],cV[cmask])
# HERE - looks like freesurface BC issues -- something is causing
# in/out fluctuations.
# 30+ m/s.
# Is it because there isn't friction here?
# there should be a lot of flow out -- eta is piling up,
# it's fairly steady in time, uniform over layers.
# so why is there inflow in the shallow cells?
# maybe watch these edges?

##


def figure_edge(ds,j,num=2):
    plt.figure(num).clf()
    fig,(ax_c1,ax_j,ax_c2) = plt.subplots(1,3,num=num,sharey=True)

    c1,c2=g.edge_to_cells(j)

    t=-1

    z_min=np.inf
    z_max=-np.inf


    Nk=ds.dims['Nk']
    for c,ax in [ (c1,ax_c1),
                  (c2,ax_c2) ]:
        ax.set_title('Cell %d'%c)
        if c<0:
            continue
        z_bed= -ds.dv.values[c]
        ax.axhline(z_bed, color='k', label='bed')
        eta=ds.eta.isel(time=t).values[c]
        ax.axhline( eta, color='k',label='eta')
        z_min=min(z_min,z_bed)
        z_max=max(z_max,eta)

        # velocity magnitude at z-layer mid points
        for kk in range(Nk):
            z_k = -ds.z_r.values[kk]
            ax.text(0,z_k,"[k=%d] |u|=%.2f"%(kk,umag.values[kk,c]),fontsize=6,
                    ha='center',va='center',clip_on=True)

        for kk in range(Nk+1): # vertical interfaces
            z_w_k = -ds.z_w.values[kk]
            ax.axhline(z_w_k,color='k',lw=0.5,alpha=0.5)
            w=float(ds.w.isel(time=t,Nc=c,Nkw=kk).values)
            ax.text(0,z_w_k,"w=%.3f"%w,color='blue',fontsize=6,ha='center',
                    va='center',clip_on=True)

        ax.axis(xmin=-0.5,xmax=0.5)

    # edge velocities
    for kk in range(Nk):
        z_k = -ds.z_r.values[kk]
        u_j = ds.U.isel(time=t).values[kk,j]
        ax_j.text(0,z_k,"[k=%d] U=%.2f"%(kk,u_j),fontsize=6,
                  ha='center',va='center')
    ax_j.axis(xmin=-0.5,xmax=0.5)
    ax_j.set_title('Edge %d'%j)

    ax_c1.axis(ymin=z_min-0.5,ymax=z_max+0.5)

# is it surprising that the edge normal goes from
# c2 to c1?

# shouldn't matter -- python doesn't write any edge normals out.
# peculiar that the two layers of this edge have the same velocity


figure_edge(ds,13333,2)

##

# 13333 - a short old river boundary edge.
#  inside cell is deep enough -
#  if the edge velocity, 13334, is used to come up with a cell centered
#  velocity, and that is in turn used to drive advection, we could have
#  a problem.
# bc_ds specifies h and uc/vc for these cells.
#  1334 is marked 3, along with the other boundaries of that cell.
#  land boundaries are 1.
# pretty sure velocity boundaries are 2.
# so we should be setting marker 3 edges to 0 velocity.
# 


##

# Watching the internal edge between the two type3 cells.
 D[j] ends up 2.51053

[p=0 j=13333 k= 19] after implicit vertical solve utmp=-23.018905421
[p=0 j=13333 k= 20] after implicit vertical solve utmp=-22.518691762
[p=0 j=13333 k= 21] after implicit vertical solve utmp=-21.483828312
[p=0 j=13333 k= 22] after implicit vertical solve utmp=-20.242323033
[p=0 j=13333 k= 23] after implicit vertical solve utmp=-18.375296211
[p=0 j=13333 k= 24] after implicit vertical solve utmp=-13.726812805
[p=0 j=13333 k= 25] after implicit vertical solve utmp=-0.586257385
[p=0 c=4130] end of UpdateDZ, ctop=19  ctopold=19
[p=0 c=4130 k= 0]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 1]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 2]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 3]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 4]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 5]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 6]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 7]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 8]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 9]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 10]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 11]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 12]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 13]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 14]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 15]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 16]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 17]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 18]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 19]   dzz=0.24796  dzzold=0.24796
[p=0 c=4130 k= 20]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 21]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 22]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 23]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 24]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 25]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 26]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 27]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 28]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 29]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 30]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 31]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 32]   dzz=0.25925  dzzold=0.25925
Time step 34255, CGSolve free-surface converged after 17 iterations, res=9.196884e-11 < 1.00e-10
[p=0 j=13333 k= 19] implicit fs delta=-0.000000000  u=-23.018905421 E=1.00000
[p=0 j=13333 k= 20] implicit fs delta=-0.000000000  u=-22.518691762 E=1.00000
[p=0 j=13333 k= 21] implicit fs delta=-0.000000000  u=-21.483828312 E=1.00000
[p=0 j=13333 k= 22] implicit fs delta=-0.000000000  u=-20.242323033 E=1.00000
[p=0 j=13333 k= 23] implicit fs delta=-0.000000000  u=-18.375296211 E=1.00000
[p=0 j=13333 k= 24] implicit fs delta=-0.000000000  u=-13.726812805 E=1.00000
[p=0 j=13333 k= 25] implicit fs delta=-0.000000000  u=-0.586257385 E=0.99776

UPredictor edge velocity profile just after implicit barotropic
[p=0 j=13333 k= 19] etop=19
[p=0 j=13333 k= 19] u=-23.018905421
[p=0 j=13333 k= 20] u=-22.518691762
[p=0 j=13333 k= 21] u=-21.483828312
[p=0 j=13333 k= 22] u=-20.242323033
[p=0 j=13333 k= 23] u=-18.375296211
[p=0 j=13333 k= 24] u=-13.726812805
[p=0 j=13333 k= 25] u=-0.586257385
[p=0 c=4130] end of UpdateDZ, ctop=19  ctopold=19
[p=0 c=4130 k= 0]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 1]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 2]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 3]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 4]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 5]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 6]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 7]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 8]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 9]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 10]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 11]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 12]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 13]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 14]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 15]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 16]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 17]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 18]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 19]   dzz=0.24796  dzzold=0.24796
[p=0 c=4130 k= 20]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 21]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 22]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 23]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 24]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 25]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 26]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 27]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 28]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 29]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 30]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 31]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 32]   dzz=0.25925  dzzold=0.25925

utmp update due to old terms, old nonhydro
[p=0 j=13333 k= 19] u=-23.018905421  utmp - u=0.000000000
[p=0 j=13333 k= 20] u=-22.518691762  utmp - u=0.000000000
[p=0 j=13333 k= 21] u=-21.483828312  utmp - u=0.000000000
[p=0 j=13333 k= 22] u=-20.242323033  utmp - u=0.000000000
[p=0 j=13333 k= 23] u=-18.375296211  utmp - u=0.000000000
[p=0 j=13333 k= 24] u=-13.726812805  utmp - u=0.000000000
[p=0 j=13333 k= 25] u=-0.586257385  utmp - u=0.000000000

Cn_U from coriolis and baroclinicity
[p=0 j=13333 k= 19] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 20] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 21] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 22] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 23] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 24] Cn_U=0.000000000 m/s
[p=0 j=13333 k= 25] Cn_U=0.000000000 m/s

Cn_U updated from cell 4129
  [p=0 j=13333 k= 19 c= 4129] delta Cn_U=-0.000101354  stmp=0.00103, stmp2=-0.00055
  [p=0 j=13333 k= 20 c= 4129] delta Cn_U=-0.000098241  stmp=0.00102, stmp2=-0.00055
  [p=0 j=13333 k= 21 c= 4129] delta Cn_U=-0.000093764  stmp=0.00099, stmp2=-0.00055
  [p=0 j=13333 k= 22 c= 4129] delta Cn_U=-0.000089130  stmp=0.00096, stmp2=-0.00055
  [p=0 j=13333 k= 23 c= 4129] delta Cn_U=-0.000084274  stmp=0.00093, stmp2=-0.00054
  [p=0 j=13333 k= 24 c= 4129] delta Cn_U=-0.000079084  stmp=0.00089, stmp2=-0.00053
  [p=0 j=13333 k= 25 c= 4129] delta Cn_U=-0.000072786  stmp=0.00079, stmp2=-0.00045

Cn_U updated from cell 4130
  [p=0 j=13333 k= 19 c= 4130] delta Cn_U=0.000140263   stmp=0.00133, stmp2=-0.00066
  [p=0 j=13333 k= 20 c= 4130] delta Cn_U=0.000135195   stmp=0.00130, stmp2=-0.00065
  [p=0 j=13333 k= 21 c= 4130] delta Cn_U=0.000128082   stmp=0.00126, stmp2=-0.00064
  [p=0 j=13333 k= 22 c= 4130] delta Cn_U=0.000121067   stmp=0.00121, stmp2=-0.00063
  [p=0 j=13333 k= 23 c= 4130] delta Cn_U=0.000114378   stmp=0.00117, stmp2=-0.00062
  [p=0 j=13333 k= 24 c= 4130] delta Cn_U=0.000108540   stmp=0.00112, stmp2=-0.00060
  [p=0 j=13333 k= 25 c= 4130] delta Cn_U=0.000110031   stmp=0.00108, stmp2=-0.00055
  [p=0 j=13333 k= 26 c= 4130] delta Cn_U=0.000081721   stmp=0.00093, stmp2=-0.00055
  [p=0 j=13333 k= 27 c= 4130] delta Cn_U=0.000026552   stmp=0.00069, stmp2=-0.00062
  [p=0 j=13333 k= 28 c= 4130] delta Cn_U=0.000045845   stmp=0.00063, stmp2=-0.00044
  [p=0 j=13333 k= 29 c= 4130] delta Cn_U=0.000044808   stmp=0.00062, stmp2=-0.00043
  [p=0 j=13333 k= 30 c= 4130] delta Cn_U=0.000043630   stmp=0.00060, stmp2=-0.00042
  [p=0 j=13333 k= 31 c= 4130] delta Cn_U=0.000041904   stmp=0.00058, stmp2=-0.00040
  [p=0 j=13333 k= 32 c= 4130] delta Cn_U=0.000038023   stmp=0.00053, stmp2=-0.00037

utmp update due to rest of HorizontalSource
[p=0 j=13333 k= 19] fab1*Cn_U=-0.001005379
[p=0 j=13333 k= 20] fab1*Cn_U=-0.000970699
[p=0 j=13333 k= 21] fab1*Cn_U=-0.000921702
[p=0 j=13333 k= 22] fab1*Cn_U=-0.000872724
[p=0 j=13333 k= 23] fab1*Cn_U=-0.000824709
[p=0 j=13333 k= 24] fab1*Cn_U=-0.000779952
[p=0 j=13333 k= 25] fab1*Cn_U=-0.000768514

explicit freesurface addition to utmp
[p=0 j=13333 k= 19] delta utmp=-0.000000000
[p=0 j=13333 k= 20] delta utmp=-0.000000000
[p=0 j=13333 k= 21] delta utmp=-0.000000000
[p=0 j=13333 k= 22] delta utmp=-0.000000000
[p=0 j=13333 k= 23] delta utmp=-0.000000000
[p=0 j=13333 k= 24] delta utmp=-0.000000000
[p=0 j=13333 k= 25] delta utmp=-0.000000000
[p=0 j=13333] interpolated eddy viscosity
   k= 20    nu_tv=c=0.00000
   k= 21    nu_tv=c=0.00000
   k= 22    nu_tv=c=0.00000
   k= 23    nu_tv=c=0.00000
   k= 24    nu_tv=c=0.00000
   k= 25    nu_tv=c=0.00000
[p=0 j=13333 k= 20] explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 21] explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 22] explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 23] explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 24] explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 19] top explicit vertical adv delta utmp=0.000000000
[p=0 j=13333 k= 25] bot explicit vertical adv delta utmp=0.000000000
[p=0 j=13333] tri-diagonal system before vertical adv of u-mom
  1.0001 -0.0001 0.0000 0.0000 0.0000 0.0000 0.0000  u[k= 19] = d -23.019910800
 -0.0000  1.0001 -0.0000 0.0000 0.0000 0.0000 0.0000  u[k= 20] = d -22.519662461
 0.0000 -0.0000  1.0001 -0.0000 0.0000 0.0000 0.0000  u[k= 21] = d -21.484750014
 0.0000 0.0000 -0.0000  1.0001 -0.0000 0.0000 0.0000  u[k= 22] = d -20.243195757
 0.0000 0.0000 0.0000 -0.0000  1.0001 -0.0000 0.0000  u[k= 23] = d -18.376120921
 0.0000 0.0000 0.0000 0.0000 -0.0000  1.0001 -0.0000  u[k= 24] = d -13.727592757
 0.0000 0.0000 0.0000 0.0000 0.0000 -0.0000  1.0023  u[k= 25] = d -0.587025899

[p=0 j=13333] CdT=0.0000 CdB=0.0025
[p=0 j=13333] tri-diagonal system
  1.0001 -0.0001 0.0000 0.0000 0.0000 0.0000 0.0000  u[k= 19] = utmp -23.019910800  d -23.019910800
 -0.0000  1.0001 -0.0000 0.0000 0.0000 0.0000 0.0000  u[k= 20] = utmp -22.519662461  d -22.519662461
 0.0000 -0.0000  1.0001 -0.0000 0.0000 0.0000 0.0000  u[k= 21] = utmp -21.484750014  d -21.484750014
 0.0000 0.0000 -0.0000  1.0001 -0.0000 0.0000 0.0000  u[k= 22] = utmp -20.243195757  d -20.243195757
 0.0000 0.0000 0.0000 -0.0000  1.0001 -0.0000 0.0000  u[k= 23] = utmp -18.376120921  d -18.376120921
 0.0000 0.0000 0.0000 0.0000 -0.0000  1.0001 -0.0000  u[k= 24] = utmp -13.727592757  d -13.727592757
 0.0000 0.0000 0.0000 0.0000 0.0000 -0.0000  1.0023  u[k= 25] = utmp -0.587025899  d -0.587025899

[p=0 j=13333] tri-diagonal system output
  utmp[k= 19]=-23.019879780  E[k= 19]=1.0000  dzf[k= 19]=0.24796
  utmp[k= 20]=-22.519649619  E[k= 20]=1.0000  dzf[k= 20]=0.40240
  utmp[k= 21]=-21.484743633  E[k= 21]=1.0000  dzf[k= 21]=0.40240
  utmp[k= 22]=-20.243176440  E[k= 22]=1.0000  dzf[k= 22]=0.40240
  utmp[k= 23]=-18.376035028  E[k= 23]=1.0000  dzf[k= 23]=0.40240
  utmp[k= 24]=-13.727288439  E[k= 24]=1.0000  dzf[k= 24]=0.40240
  utmp[k= 25]=-0.586262647  E[k= 25]=0.9978  dzf[k= 25]=0.25114
   D[j] ends up 2.51053

[p=0 j=13333 k= 19] after implicit vertical solve utmp=-23.019879780
[p=0 j=13333 k= 20] after implicit vertical solve utmp=-22.519649619
[p=0 j=13333 k= 21] after implicit vertical solve utmp=-21.484743633
[p=0 j=13333 k= 22] after implicit vertical solve utmp=-20.243176440
[p=0 j=13333 k= 23] after implicit vertical solve utmp=-18.376035028
[p=0 j=13333 k= 24] after implicit vertical solve utmp=-13.727288439
[p=0 j=13333 k= 25] after implicit vertical solve utmp=-0.586262647
[p=0 c=4130] end of UpdateDZ, ctop=19  ctopold=19
[p=0 c=4130 k= 0]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 1]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 2]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 3]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 4]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 5]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 6]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 7]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 8]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 9]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 10]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 11]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 12]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 13]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 14]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 15]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 16]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 17]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 18]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 19]   dzz=0.24796  dzzold=0.24796
[p=0 c=4130 k= 20]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 21]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 22]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 23]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 24]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 25]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 26]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 27]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 28]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 29]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 30]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 31]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 32]   dzz=0.25925  dzzold=0.25925
Time step 34256, CGSolve free-surface converged after 17 iterations, res=9.196917e-11 < 1.00e-10
[p=0 j=13333 k= 19] implicit fs delta=-0.000000000  u=-23.019879780 E=1.00000
[p=0 j=13333 k= 20] implicit fs delta=-0.000000000  u=-22.519649619 E=1.00000
[p=0 j=13333 k= 21] implicit fs delta=-0.000000000  u=-21.484743633 E=1.00000
[p=0 j=13333 k= 22] implicit fs delta=-0.000000000  u=-20.243176440 E=1.00000
[p=0 j=13333 k= 23] implicit fs delta=-0.000000000  u=-18.376035028 E=1.00000
[p=0 j=13333 k= 24] implicit fs delta=-0.000000000  u=-13.727288439 E=1.00000
[p=0 j=13333 k= 25] implicit fs delta=-0.000000000  u=-0.586262647 E=0.99776

UPredictor edge velocity profile just after implicit barotropic
[p=0 j=13333 k= 19] etop=19
[p=0 j=13333 k= 19] u=-23.019879780
[p=0 j=13333 k= 20] u=-22.519649619
[p=0 j=13333 k= 21] u=-21.484743633
[p=0 j=13333 k= 22] u=-20.243176440
[p=0 j=13333 k= 23] u=-18.376035028
[p=0 j=13333 k= 24] u=-13.727288439
[p=0 j=13333 k= 25] u=-0.586262647
[p=0 c=4130] end of UpdateDZ, ctop=19  ctopold=19
[p=0 c=4130 k= 0]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 1]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 2]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 3]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 4]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 5]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 6]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 7]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 8]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 9]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 10]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 11]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 12]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 13]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 14]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 15]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 16]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 17]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 18]   dzz=0.00000  dzzold=0.00000
[p=0 c=4130 k= 19]   dzz=0.24796  dzzold=0.24796
[p=0 c=4130 k= 20]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 21]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 22]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 23]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 24]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 25]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 26]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 27]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 28]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 29]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 30]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 31]   dzz=0.40240  dzzold=0.40240
[p=0 c=4130 k= 32]   dzz=0.25925  dzzold=0.25925
----------------------------------------------------------------------
Time step 34256: Processor 0, Run is blowing up!
Horizontal Courant number problems:
  Grid indices: j=13339 k=19 (Nke=33)
!!Error with pointer => h[-1]!!
  Location: x=6.4305622e+05, y=4.1872811e+06, z=-4.462e+00
  Free-surface heights (on either side): -7.800e+00, -7.800e+00
  Depths (on either side): 1.314e+01, 1.314e+01
  Flux-face height = 2.480e-01, CdB = 2.500e-03
  Umax = -3.663e+01
  Horizontal grid spacing grid->dg[13339] = 1.831e+01
  Horizontal Courant number is CmaxU = 1.00.
  You specified a maximum of 1.00 in suntans.dat
  Your time step size is 0.50.
  Reducing it to at most 0.25 (For C=0.5) might solve this problem.
----------------------------------------------------------------------
Outputting blowup data to netcdf at step 34256 of 43200
time start: 58
SYNCING netcdf
SYNCING netcdf
SYNCING netcdf
SYNCING netcdf
