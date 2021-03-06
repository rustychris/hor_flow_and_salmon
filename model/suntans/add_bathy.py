"""
Add bathy data to grid.
Had been in the grid directory, but moved here for quicker edit-run cycles
"""
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp
import numpy as np
import os

utils.path(os.path.join(os.path.dirname(__file__),
                        "../../bathy"))
import bathy

def add_bathy(g,suffix=''):
    dem=bathy.dem(suffix=suffix)

    z=dem(g.cells_center())
    z[np.isnan(z)]=5.0
    g.add_cell_field('z_bed', z, on_exists='overwrite')

    return g

def postprocess(g,suffix=""):
    if suffix=="":
        pass
    elif suffix=='med0':
        L=35
        
        # First cut at anisotropic median filter.
        # depends on the grid matching this set of BCs:
        # see smooth_original_aniso.py for dev details
        print("______________________MED0___________________")
        bcs=wkb2shp.shp2geom("../grid/snubby_junction/forcing-snubby-01.shp")
        
        OR_left =np.array(bcs['geom'][1].coords)[0] # left end at OR
        OR_right=np.array(bcs['geom'][1].coords)[-1] # right end at OR
        SJ_up_left=np.array(bcs['geom'][0].coords)[0]
        SJ_up_right=np.array(bcs['geom'][0].coords)[-1]
        SJ_dn_right=np.array(bcs['geom'][2].coords)[-1]
        SJ_dn_left=np.array(bcs['geom'][2].coords)[0]

        # nodes on boundaries:
        river_left = g.select_nodes_boundary_segment( coords=[OR_left,SJ_up_left] ) 
        river_right= g.select_nodes_boundary_segment( coords=[SJ_up_right,SJ_dn_right] )
        river_split= g.select_nodes_boundary_segment( coords=[SJ_dn_left,OR_right] )

        # cells on boundaries:
        river_left_cells =np.unique( [ c for n in river_left for c in g.node_to_cells(n) ] )
        river_right_cells=np.unique( [ c for n in river_right for c in g.node_to_cells(n) ] )
        river_split_cells=np.unique( [ c for n in river_split for c in g.node_to_cells(n) ] )

        # Solve a Laplace equation to get stream function
        from stompy.model import unstructured_diffuser
        diff=unstructured_diffuser.Diffuser(g)

        for c in river_left_cells:
            diff.set_dirichlet(0,cell=c)

        for c in river_right_cells:
            diff.set_dirichlet(100,cell=c)

        for c in river_split_cells:
            diff.set_dirichlet(50,cell=c)

        diff.construct_linear_system()
        diff.solve_linear_system(animate=False)

        # Stream function on the cell centers
        psi=diff.C_solved

        # do the smoothing on the grid itself...
        cc=g.cells_center()
        d_orig=g.cells['z_bed']

        d_med=d_orig.copy()

        for c in utils.progress(g.valid_cell_iter()):
            # Get a large-ish neighborhood:
            #nbrs=np.array(g.select_cells_nearest( cc[c], count=200))
            nbrs=np.nonzero( utils.mag(cc-cc[c]) < L)[0]

            alpha=10 # controls the degree of anisotropy
            coords=np.c_[ cc[nbrs], alpha*psi[nbrs]]
            coord0=np.array( [ cc[c,0],cc[c,1],alpha*psi[c] ] )

            dists=utils.mag(coords-coord0)

            # Will take the median of a subset of those:
            # again, scale by size of nbrs to be grid-invariant
            N=int(0.15*len(nbrs))
            subsel=np.argsort(dists)[:N]
            close_nbrs=nbrs[subsel]
            d_med[c] = np.median( d_orig[close_nbrs] )

        g.cells['z_bed']=d_med
    else:
        assert False,"What is post processing suffix %s"%suffix
