"""
Created on Fri 24 Mar 2018

@author: Ed Gross, Steve Andrews
@organization: Resource Management Associates
@contact: ed@rmanet.com, steve@rmanet.com
@note: Highly simplified river model. Current assumptions:
    - unsteady and barotropic terms only
    - theta = 1
    - no subgrid
    - circumcenters inside cell
"""
from __future__ import print_function
import six
import os

import random
import time

import h5py
from matplotlib.collections import PolyCollection
from scipy.spatial.distance import euclidean
from stompy import utils
from stompy.grid import unstructured_grid as ugrid

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

# global variables
g = 9.8  # gravity
dc_min = 1.0
max_sides = 8
dzmin = 0.001

class SwampyCore(object):
    # tolerance for conjugate gradient
    cg_tol=1.e-12

    def __init__(self,dt=0.025,theta=1.0,ManN=0.0):
        self.dt = dt
        self.bcs=[]
        self.theta = theta
        self.ManN = ManN
        # Default advection
        self.get_fu=self.get_fu_Perot

    def add_bc(self,bc):
        self.bcs.append(bc)

    def center_vel_perot(self, i, cell_uj):
        """
        Cell center velocity calculated with Perot dual grid interpolation
        return u center veloctiy vector, {x,y} components, based on
        edge-normal velocities in cell_uj.
        """
        acu = np.zeros(2, np.float64)
        uside = np.zeros_like(acu)
        ucen = np.zeros_like(acu)
        nsides = len(cell_uj)
        for l in range(nsides):
            j = self.grd.cells[i]['edges'][l]
            for dim in range(2):  # x and y
                uside[dim] = cell_uj[l] * self.en[j, dim]  # u vel component of normal vel
                acu[dim] += uside[dim] * self.len[j] * self.dist[i, l]
        for dim in range(2):  # x and y
            ucen[dim] = acu[dim] / self.grd.cells['_area'][i]

        return ucen

    def get_center_vel(self, uj):
        """
        Get center velocities for every cell in the grid
        """
        ui = np.zeros((self.ncells, 2), np.float64)
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            cell_uj=uj[ self.grd.cells['edges'][i,:nsides] ]
            ui[i] = self.center_vel_perot(i, cell_uj)
            
        return ui

    def get_fu_no_adv(self, uj, fu, **kw):
        """
        Return fu term with no advection
        """
        # allocating fu and copying uj has been moved up and out of the advection
        # methods
        return
    
    def get_fu_orig(self, uj, fu, **kw):
        """
        Add to fu term the original advection method.
        """
        ui = self.get_center_vel(uj)
        for j in self.intern:  # loop over internal cells
            ii = self.grd.edges[j]['cells']  # 2 cells
            if uj[j] > 0.:
                i_up = ii[0]
            elif uj[j] < 0.:
                i_up = ii[1]
            else:
                fu[j] = 0.0
                continue
            # ui_norm = component of upwind cell center velocity perpendicular to cell face
            ui_norm = ui[i_up, 0] * self.en[j, 0] + ui[i_up, 1] * self.en[j, 1]
            # this has been updated to just add to fu, since fu already has explicit barotropic
            # and uj[j]
            fu[j] += uj[j] * (-2.*self.dt * np.sign(uj[j]) * (uj[j] - ui_norm) / self.dc[j])

        return fu

    def add_explicit_barotropic(self,hjstar,hjbar,hjtilde,ei,fu):
        """
        return the per-edge, explicit baroptropic acceleration.
        This used to be part of get_fu_Perot, but is split out since it is not 
        specifically advection.

        hjbar, hjtilde: Kramer and Stelling edge heights
        ei: cell-centered freesurface
        dest: edge-centered velocity array to which the term is added, typ. fu
        """
        if self.theta==1.0: # no explicit term
            return

        # three cases: deep enough to have the hjbar term, wet but not deep,
        # and dry.
        deep=hjbar>=dzmin
        dry=hjstar<=0
        
        hterm=np.ones_like(hjbar)
        hterm[deep]=hjtilde[deep]/hjbar[deep]
        hterm[dry]=0.0
        
        gDtThetaCompl = g * self.dt * (1.0 - self.theta)
        iLs=self.grd.edges['cells'][:,0]
        iRs=self.grd.edges['cells'][:,1]
        fu[self.intern] -= (gDtThetaCompl * hterm * (ei[iRs] - ei[iLs]) / self.dc[:])[self.intern]
        
    def get_fu_Perot(self, uj, alpha, sil, hi, hjstar, hjbar, hjtilde, fu):
        """
        Return fu term with Perot method.
        """
        ui = self.get_center_vel(uj)

        # some parts can be vectorized
        iLs=self.grd.edges['cells'][:,0]
        iRs=self.grd.edges['cells'][:,1]

        lhu=self.len[:] * hjstar[:] * uj[:]

        cell_area=self.grd.cells['_area']

        for j in self.intern:
            if hjstar[j]<=0: continue # skip dry edges
            
            # ii = self.grd.edges[j]['cells']
            iL = iLs[j]
            iR = iRs[j]

            # explicit barotropic moved out to separate function
            
            # left cell
            nsides = self.ncsides[iL]
            sum1 = 0.
            for l in range(nsides):
                k = self.grd.cells[iL]['edges'][l]
                Q = sil[iL, l] * lhu[k] # lhu[k]==self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                # i2 = iitmp[np.nonzero(iitmp - iL)[0][0]]  # get neighbor
                i2 = iitmp[self.sil_idx[iL,l]]
                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])

            fu[j] -= self.dt * alpha[j, 0] * sum1 / (cell_area[iL] * hjbar[j])

            # right cell
            sum1 = 0.
            nsides = self.ncsides[iR]
            for l in range(nsides):
                k = self.grd.cells[iR]['edges'][l]
                Q = sil[iR, l] * lhu[k] # lhu[k]==self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                # i2 = iitmp[np.nonzero(iitmp - iR)[0][0]]  # get neighbor
                i2 = iitmp[self.sil_idx[iR,l]]

                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])
            fu[j] -= self.dt * alpha[j, 1] * sum1 / (cell_area[iR] * hjbar[j])

        return fu

    def calc_phi(r, limiter):
        """
        Calculate limiter phi
        """
        if limiter == 'vanLeer':
            phi = (r + abs(r)) / (1.0 + r)
        elif limiter == 'minmod':
            phi = max(0.0, min(r, 1.0))
        else:
            print('limiter not defined %s' % limiter)

        return phi

    def calc_hjstar(self, ei, zi, uj):
        """
        Calculate hjstar from Kramer and Stelling
        """
        hjstar = np.zeros(self.nedges)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']

            if uj[j] > 0:
                e_up = ei[ii[0]]
            elif uj[j] < 0:
                e_up = ei[ii[1]]
            else:
                e_up = max(ei[ii[0]], ei[ii[1]])
            zj=min(zi[ii[0]], zi[ii[1]])
            # force non-negative here.
            # it is possible for zj+e_up<0.  this is not an error, just
            # an aspect of the discretization.
            # nonetheless hjstar should not be allowed to go negative
            hjstar[j] = max(0, zj + e_up)

        return hjstar

    def calc_hjbar(self, ei, zi):
        """
        Calculate hjbar from Kramer and Stelling
        (linear interpolation from depth in adjacent cells)
        """
        hjbar = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            # Original code recalculated.
            # RH  - hi is already limited to non-negative
            hL=self.hi[ii[0]]
            hR=self.hi[ii[1]]
            hjbar[j] = self.alpha[j][0] * hL + self.alpha[j][1] * hR

        assert np.all(hjbar>=0)
        return hjbar

    def calc_hjtilde(self, ei, zi):
        """
        Calculate hjtilde from Kramer and Stelling
        (with average cell eta, and interpolated cell bed)
        """
        hjtilde = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            eavg = 0.5 * (ei[ii[0]] + ei[ii[1]])
            bL = zi[ii[0]]
            bR = zi[ii[1]]
            hjtilde[j] = eavg + self.alpha[j][0] * bL + self.alpha[j][1] * bR
        if np.any(hjtilde<0):
            print("%d/%d hjtilde<0"%( (hjtilde<0).sum(), len(hjtilde)))
        return hjtilde

    def calc_volume(self, hi):
        """
        Calculate cell volume give cell total water depth
        """
        #vol = np.zeros(self.ncells)
        #for i in range(self.ncells):
        #    vol[i] = self.grd.cells['_area'][i] * hi[i]
        #return vol
        return self.grd.cells['_area'][:] * hi[:]

    def calc_wetarea(self, hi):
        """
        Calculate cell wetted area give cell total water depth
        """
        # RH: check had been disabled -- for speed, or correctness??
        return np.where( hi>0.000001,
                         self.grd.cells['_area'],
                         0.0 )

    def calc_edge_wetarea(self, hj):
        """
        Calculate wetted area for the cell faces
        """
        Aj=self.len * hj
        assert np.all(Aj>=0)
        return Aj

    def calc_edge_friction(self, uj, aj, hj):
        """
        Calculate friction coef at edge
        Cf = n^2*g*|u|/Rh^(4/3)
        """
        cf = np.zeros(self.nedges)
        n = self.ManN
        for j in self.intern:
            rh = hj[j]  # assume no side wall friction
            if rh < dzmin:
                cf[j] = 0.
            else:
                cf[j] = n * n * g * np.abs(uj[j]) / rh ** (4. / 3.)
        return cf

    def set_grid(self,ug):
        """
        Set grid from unstructured_grid.UnstructuredGrid instance
        """
        self.grd=ug
        self.set_topology()

    def set_topology(self):
        """
        Use functions of unstructured grid class for remaining topology
        """
        self.nedges = self.grd.Nedges()
        self.ncells = self.grd.Ncells()
        self.nnodes = self.grd.Nnodes()
        self.grd.update_cell_edges()
        self.grd.update_cell_nodes()
        self.grd.edge_to_cells()
        self.grd.cells_area()
        self.grd.cells['_center'] = self.grd.cells_centroid()
        self.grd.edges['mark'] = 0  # default is internal cell
        self.extern = np.where(np.min(self.grd.edges['cells'], axis=1) < 0)[0]
        self.grd.edges['mark'][self.extern] = 1  # boundary edge
        self.intern = np.where(self.grd.edges['mark'] == 0)[0]
        self.nedges_intern = len(self.intern)  # number of internal edges
        self.exy = self.grd.edges_center()
        self.en = self.grd.edges_normals()
        self.len = self.grd.edges_length()
        # number of valid sides for each cell
        self.ncsides = np.asarray([sum(jj >= 0) for jj in self.grd.cells['edges']])

        return

    def get_cell_center_spacings(self):
        """
        Return cell center spacings
        Spacings for external edges are set to dc_min (should not be used)
        """
        dc = dc_min * np.ones(self.nedges, np.float64)
        if 0:
            for j in self.intern:
                ii = self.grd.edges[j]['cells']  # 2 cells
                xy = self.grd.cells[ii]['_center']  # 2x2 array
                dc[j] = euclidean(xy[0], xy[1])  # from scipy
        else:
            # vectorize
            i0=self.grd.edges['cells'][self.intern,0]
            i1=self.grd.edges['cells'][self.intern,1]
            dc[self.intern] = utils.dist( self.grd.cells['_center'][i0],
                                          self.grd.cells['_center'][i1] )
        self.dc = dc

        return dc

    def edge_to_cen_dist(self):
        dist = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            cen_xy = self.grd.cells['_center'][i]
            nsides = self.ncsides[i]
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                side_xy = self.exy[j]
                # if grid is not 'cyclic', i.e. all nodes fall on circumcircle,
                # then this is not accurate, and according to Steve's code
                # it is necessary to actually intersect the line between cell
                # centers with the edge.
                # not sure that it wouldn't be better to use the perpendicular
                # distance, though.
                dist[i, l] = utils.dist(side_xy, cen_xy) # faster

        return dist

    def get_alphas_Perot(self):
        """
        Return Perot weighting coefficients
        Coefficients for external edges are set to zero (should not be used)
        """
        if 0:
            alpha = np.zeros((self.nedges, 2), np.float64)
            for j in self.intern:
                side_xy = self.exy[j]
                ii = self.grd.edges[j]['cells']
                cen_xyL = self.grd.cells['_center'][ii[0]]
                cen_xyR = self.grd.cells['_center'][ii[1]]
                alpha[j, 0] = euclidean(side_xy, cen_xyL) / self.dc[j]
                alpha[j, 1] = euclidean(side_xy, cen_xyR) / self.dc[j]
            self.alpha = alpha
        else: # vectorized
            # should probably be refactored to used edge-center distances as above.
            # for cyclic grid should be fine.  for non-cyclic, need to verify what
            # the appropriate distance (center-to-intersection, or perpendicular)
            alpha = np.zeros((self.nedges, 2), np.float64)
            i0=self.grd.edges['cells'][self.intern,0]
            i1=self.grd.edges['cells'][self.intern,1]
            cen_xyL = self.grd.cells['_center'][i0]
            cen_xyR = self.grd.cells['_center'][i1]
            alpha[self.intern, 0] = utils.dist(self.exy[self.intern], cen_xyL) / self.dc[self.intern]
            alpha[self.intern, 1] = utils.dist(self.exy[self.intern], cen_xyR) / self.dc[self.intern]
            self.alpha = alpha

        return alpha

    def get_sign_array(self):
        """
        Sign array for all cells, for all edges
        Positive if face normal is pointing out of cell
        """
        sil = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                ii = self.grd.edges[j]['cells']  # 2 cells
                sil[i, l] = (ii[1] - 2 * i + ii[0]) / (ii[1] - ii[0])

        # for indexing -- just 0,1
        self.sil_idx = ((sil+1)//2).astype(np.int64)

        return sil

    def get_side_num_of_cell(self):
        """
        Sparse matrix of local side number (0,1,2,or 3) for each cell, for each edge
        """
        lij = -1 * np.ones((self.ncells, self.nedges), np.int32)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                lij[i, j] = l

        return lij

    def set_initial_conditions(self):
        """
        Subclasses or caller can specialize this.  Here we just allocate
        the arrays
        """
        # Initial free surface elevation, positive up
        self.ic_ei = np.zeros(self.ncells, np.float64)
        # Bed elevation, positive down
        self.ic_zi = np.zeros_like(self.ic_ei)

        # other contenders:
        # self.ds_i  index array for cell BCs on downstream end
        # self.us_i  index array for cell BCs on upstream end
        # self.us_j  index array for edge BCs on upstream end
        # need to refactor BC code

    def run(self, tend):
        """
        Run simulation
        """
        dt = self.dt
        self.tend = tend
        self.t=0.0 # for now, always start at 0.
        nsteps = np.int(np.round(tend / dt))

        # precompute constants
        gdt2theta2 = g * dt * dt * self.theta * self.theta

        dc = self.get_cell_center_spacings()
        self.dist = self.edge_to_cen_dist()
        alpha = self.get_alphas_Perot()
        sil = self.get_sign_array()
        # lij = self.get_side_num_of_cell()

        # edge length divided by center spacing
        len_dc_ratio = np.divide(self.len, dc)

        # cell center values
        ncells = self.ncells
        self.hi = hi = np.zeros(ncells, np.float64)  # total depth
        self.zi = zi = np.zeros_like(hi)  # bed elevation, measured positive down
        self.ei = ei = np.zeros_like(hi)  # water surface elevation
        self.vi = vi = np.zeros_like(hi)  # cell volumes
        self.pi = pi = np.zeros_like(hi)  # cell wetted areas

        # edge values
        self.uj = uj = np.zeros(self.nedges, np.float64)  # normal velocity at side
        self.qj = qj = np.zeros(self.nedges, np.float64)  # normal velocity*h at side
        self.aj = aj = np.zeros(self.nedges, np.float64)  # edge wet areas
        cf = np.zeros(self.nedges, np.float64)  # edge friction coefs
        cfterm = np.zeros(self.nedges, np.float64)  # edge friction coefs - term for matrices

        # Matrix
        # np.zeros((ncells, ncells), np.float64)  # inner iterations
        Ai = sparse.dok_matrix( (ncells,ncells),np.float64) 
        bi = np.zeros(ncells, np.float64)
        Ao = sparse.dok_matrix( (ncells, ncells), np.float64)  # outer iterations
        bo = np.zeros(ncells, np.float64)
        x0 = np.zeros_like(hi)

        # short vars for grid
        self.area = area = self.grd.cells['_area']

        # set initial conditions
        ei[:] = self.ic_ei[:] # RH: does this need to be propped up to at least -zi?
        zi[:] = self.ic_zi[:]
        # RH
        ei[:] = np.maximum( ei,-self.zi ) 

        self.hi[:] = zi + ei # update total depth. assume ei is valid, so no clippig here.
        assert np.all(self.hi>=0.0)
                      
        hjstar = self.calc_hjstar(ei, zi, uj)
        hjbar = self.calc_hjbar(ei, zi)
        hjtilde = self.calc_hjtilde(ei, zi)
        vi[:] = self.calc_volume(hi)
        pi[:] = self.calc_wetarea(hi)
        # RH: this had been using hjbar, but that leads to issues with
        # wetting and drying. hjstar I think is more appropriate, and in
        # fact hjstar is used in the loop, just not in Ai.
        # this clears up bad updates to ei
        # a lot of this is duplicated at the end of the time loop.
        # would be better to rearrange to avoid that duplication.
        aj[:] = self.calc_edge_wetarea(hjstar)
        cf[:] = self.calc_edge_friction(uj, aj, hjbar)
        cfterm[:] = 1. / (1. + self.dt * cf[:])

        eta_cells={}
        eta_mask=np.zeros( ncells, np.bool8)
        for bc in self.bcs:
            if isinstance(bc,StageBC):
                for c in bc.cells(self):
                    eta_cells[c]=bc # doesn't matter what the value is
                    eta_mask[c]=True
        print("Total of %d stage BC cells"%len(eta_cells))

        # change code to vector operations at some point
        # time stepping loop
        tvol = np.zeros(nsteps)
        for n in range(nsteps):
            self.t=n*dt # update model time
            print('step %d/%d  t=%.3fs'%(n+1,nsteps,self.t))

            # G: explicit baroptropic term, advection term
            fu=np.zeros_like(uj)
            fu[self.intern]=uj[self.intern]
            fu[hjstar<=0]=0.0
            
            self.add_explicit_barotropic(hjstar=hjstar,hjbar=hjbar,hjtilde=hjtilde,ei=ei,fu=fu)
            # shouldn't happen, but check to be sure
            assert np.all( fu[hjstar==0.0]==0.0 )
            
            self.get_fu(uj,alpha=alpha,sil=sil,hi=hi,hjstar=hjstar,hjtilde=hjtilde,hjbar=hjbar,
                        fu=fu)
            
            # get_fu should avoid this, but check to be sure
            assert np.all( fu[hjstar==0.0]==0.0 )

            # Following Casulli 2009, eq 18
            # zeta^{m+1}=zeta^m-[P(zeta^m)+T]^{-1}[V(zeta^m)+T.zeta^m-b]
            # Rearranging...
            # [P(zeta^m)+T]  (zeta^{m+1}-zeta^m) = [V(zeta^m)+T.zeta^m-b]
            # where T holds the coefficients for the linear system.
            # note that aj goes into the T coefficients using the old freesurface
            # and is not updated as part of the linear system.
            # the matrix Ai in the code below corresponds to T 
            
            # set matrix coefficients
            # first put one on diagonal, and explicit stuff on rhs (b)
            # Ai=sparse.dok_matrix((ncells,ncells), np.float64)
            Ai_rows=[]
            Ai_cols=[]
            Ai_data=[]

            bi[:] = 0.

            for i in range(ncells):
                if i in eta_cells:
                    # Ai[i,i]=1.0
                    Ai_rows.append(i) 
                    Ai_cols.append(i)
                    Ai_data.append(1.0)
                    # RH: I think I was missing the continue before
                    continue

                sum1 = 0.0
                for l in range(self.ncsides[i]):
                    j = self.grd.cells[i]['edges'][l]
                    if self.grd.edges[j]['mark'] == 0:  # if internal
                        sum1 += sil[i, l] * aj[j] * (self.theta * cfterm[j] * fu[j] +
                                                     (1-self.theta) * uj[j])
                        if hjbar[j] > dzmin:
                            hterm = hjtilde[j] / hjbar[j]
                        else:
                            hterm = 1.0
                        coeff = gdt2theta2 * aj[j] * cfterm[j] / dc[j] * hterm
                        # Ai[i, i] += coeff
                        Ai_rows.append(i)
                        Ai_cols.append(i)
                        Ai_data.append(coeff)

                        ii = self.grd.edges[j]['cells']  # 2 cells
                        i2 = ii[np.nonzero(ii - i)[0][0]]  # i2 = index of neighbor
                        #Ai[i, i2] = -coeff
                        Ai_rows.append(i)
                        Ai_cols.append(i2)
                        Ai_data.append(-coeff)
                # RH: is this correct to have vi here?
                # Ai is formulated in terms of just the change, and does not
                # have 1's on the diagonal. don't be tempted to remove vi here.
                # it's correct.
                bi[i] = vi[i] - dt * sum1
            Ai=sparse.coo_matrix( (Ai_data,(Ai_rows,Ai_cols)), (ncells,ncells), dtype=np.float64 )

            for iter in range(10):
                if 1:
                    # This block:
                    # Ao=Ai+diag(pi)
                    Ao_rows=[] # list(Ai_rows)
                    Ao_cols=[] # list(Ai_cols)
                    Ao_data=[] # list(Ai_data)

                    #for i in range(ncells):
                    #    Ao[i, i] += pi[i]
                    non_eta=np.nonzero(~eta_mask)[0]
                    Ao_rows.extend(non_eta)
                    Ao_cols.extend(non_eta)
                    Ao_data.extend(pi[~eta_mask])
                    # Ao[~eta_mask,~eta_mask] += pi[~eta_mask]

                    Ao=sparse.coo_matrix( (Ao_data,(Ao_rows,Ao_cols)), (ncells,ncells), dtype=np.float64)
                    Ao=Ao.tocsr()
                    Ao=Ai+Ao

                #bo[:] = 0.
                #for i in range(ncells):
                #    bo[i] = vi[i] + np.dot(Ai[i, :], ei[:]) - bi[i]
                bo[:] = vi + Ai.dot(ei) - bi

                bo[eta_mask]=0. # no correction

                for bc in self.bcs:
                    if isinstance(bc,StageBC):
                        c,e,h=bc.cell_edge_eta(self)
                        ei[c]=h
                    elif isinstance(bc,FlowBC):
                        # flow bcs:
                        c,e,Q=bc.cell_edge_flow(self)
                        bo[c] += -dt * Q

                # invert matrix
                start = time.time()
                ei_corr, success = sparse.linalg.cg(Ao,
                                                    bo, x0=x0, tol=1.e-16)  # Increase tolerance for faster speed
                end = time.time()
                print('matrix solve took {0:0.4f} sec'.format(end - start))
                if success != 0:
                    raise RuntimeError('Error in convergence of conj grad solver')

                # better to clip ei and hi, not just correct hi.
                ei[:] -= ei_corr
                hi[:] = (zi + ei)
                neg=hi<0
                if np.any(neg):
                    ei[neg]=-zi[neg]
                    hi[neg]=0
                
                # Update elevations
                # for subgrid, these should become functions of eta rather than
                # hi.
                pi = self.calc_wetarea(hi)
                vi = self.calc_volume(hi)
                rms = np.sqrt(np.sum(np.multiply(ei_corr, ei_corr)) / self.ncells)
                if rms < 0.01:
                    break

            # substitute back into u solution
            for j in self.intern:  # loop over internal cells
                ii = self.grd.edges[j]['cells']  # 2 cells
                # RH: don't accelerate dry edges
                hterm = 1.0*(hjstar[j]>0)
                term = g * dt * self.theta * (ei[ii[1]] - ei[ii[0]]) / dc[j]
                uj[j] = cfterm[j] * (fu[j] - term * hterm)

            self.hjstar = hjstar = self.calc_hjstar(ei, zi, uj)
            self.hjbar = hjbar = self.calc_hjbar(ei, zi)
            self.hjtilde = hjtilde = self.calc_hjtilde(ei, zi)
            vi = self.calc_volume(hi)
            pi = self.calc_wetarea(hi)
            aj = self.calc_edge_wetarea(hjstar)
            cf = self.calc_edge_friction(uj, aj, hjbar)
            cfterm = 1. / (1. + self.dt * cf[:])

            # conservation properties
            tvol[n] = np.sum(hi * self.grd.cells['_area'][:])

            self.step_output(n=n,ei=ei,uj=uj)

        return hi, uj, tvol, ei

    def get_edge_x_vel(self, uj):
        """
        Return x direction velocity and location at cell edge midpoints
        """
        xe = np.zeros(self.nedges_intern)
        uxe = np.zeros_like(xe)
        for idx, j in enumerate(self.intern):
            xe[idx] = self.exy[j][0]  # x coordinate
            uxe[idx] = uj[j] * self.en[j][0]

        return xe, uxe

    def get_xsect_avg_val(self, eta_i):
        """
        Return cross-sectionally averaged x velocities for non-uniform grids
        """
        xc = []
        uxc = []
        nx = self.domain_length / self.dx
        for i in range(int(nx)):
            startx = float(i) * self.dx
            endx = float(i + 1) * self.dx
            midx = (float(i) + 0.5) * self.dx
            d = (self.grd.cells['_center'][:, 0] > startx) & (self.grd.cells['_center'][:, 0] < endx)
            if np.sum(d) > 0:
                utmp = np.mean(eta_i[d])
                uxc.append(utmp)
                xc.append(midx)
        return (xc, uxc)

    def get_grid_poly_collection(self):
        """
        Return list of polygon points useful for 2D plotting
        """
        xy = []
        for i in range(self.ncells):
            tmpxy = []
            for l in range(self.ncsides[i]):
                n = self.grd.cells[i]['nodes'][l]
                tmp = self.grd.nodes[n]['x']
                tmpxy.append([tmp[0], tmp[1]])
            n = self.grd.cells[i]['nodes'][0]
            tmp = self.grd.nodes[n]['x']
            tmpxy.append([tmp[0], tmp[1]])
            xy.append(tmpxy)

        return xy

    def step_output(self,n,ei,**kwargs):
        pass

# Structuring BCs:
#   for bump case, currently the code replaces downstream matrix rows
#   and rhs values with ds_eta value.
#   and upstream, sets a flow boundary in the rhs.

class BC(object):
    _cells=None
    _edges=None
    geom=None
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
    def update_matrix(self,model,A,b):
        pass
    def cells(self,model):
        if self._cells is None:
            self.set_elements(model)
        return self._cells
    def edges(self,model):
        if self._edges is None:
            self.set_elements(model)
        return self._edges
    def set_elements(self,model):
        cell_hash={}
        edges=[]
        for j in model.grd.select_edges_by_polyline(self.geom):
            c=model.grd.edges['cells'][j,0]
            if c in cell_hash:
                print("Skipping BC edge %d because its cell is already counted with a different edge"%j)
                continue
            cell_hash[c]=j
            edges.append(j)
        self._edges=np.array(edges)
        self._cells=model.grd.edges['cells'][self._edges,0]
        assert len(self._cells)==len(np.unique(self._cells))
    def plot(self,model):
        model.grd.plot_edges(color='k',lw=0.5)
        model.grd.plot_edges(color='m',lw=4,mask=self.edges(model))
        model.grd.plot_cells(color='m',alpha=0.5,mask=self.cells(model))

class FlowBC(BC):
    """
    Represents a flow in m3/s applied across a set of boundary
    edges
    """
    Q=None
    ramp_time=0.0
    def cell_edge_flow(self,model):
        """
        return a tuple: ( array of cell indices,
                          array of edge indices,
                          array of inflows, m3/s )
        """
        c=self.cells(model)
        e=self.edges(model)
        # assumes that flow boundary eta is same as interior
        # cell
        if model.t<self.ramp_time:
            ramp=model.t/self.ramp_time
        else:
            ramp=1.0

        per_edge_A=(model.len[e]*model.hi[c])
        per_edge_Q=self.Q*per_edge_A/per_edge_A.sum()
        return c,e,per_edge_Q*ramp
        
    def apply_bc(self,model,A,b):
        c,e,Q=self.cell_edge_flow(model)
        b[c] += (model.dt / model.area[c]) * Q

class StageBC(BC):
    h=None
    def apply_bc(self,model,A,b):
        c=self.cells(model)
        # changes to A are no in the loop above in SWAMPY
        #A[c, :] = 0.
        #A[c, c] = 1.
        b[c] = self.h
    def cell_edge_eta(self,model):
        c=self.cells(model)
        return c,None,self.h*np.ones_like(c)

class SWAMPpy(SwampyCore):
    def __init__(self,dt=0.025,dx=1.0):
        super(SWAMPpy,self).__init__(dt=dt)
        self.dx=dx

    def run(self,case=None,tend=None):
        # compatibility code -- case is used elsewhere to set
        # use_contract_factor.
        # and goofy default args to allow skipping case and passing tend by
        # keyword. meh.
        assert tend is not None
        super(SWAMPpy,self).run(tend=tend)
    def make_1D_grid_Cartesian(self, L, show_grid=False):
        """
        Setup unstructured grid for channel 1 cell wide
        Grid is Cartesian with edge length = self.dx
        """
        ncells = int(L / self.dx)
        npoints = 2 * ncells + 2
        nedges = 3 * ncells + 1
        points = np.zeros((npoints, 2), np.float64)
        # in future if only 3 edges, node4 will have value -1
        cells = -1 * np.ones((ncells, 4), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        for i in range(ncells + 1):  # set node x,y etc.
            points[2 * i, 0] = self.dx * i
            points[2 * i, 1] = 0.0
            points[2 * i + 1, 0] = self.dx * i
            points[2 * i + 1, 1] = self.dx
            if i < ncells:
                cells[i, :] = [2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1, 2 * i]
                # bottom of cell
                edges[3 * i, :] = [2 * (i + 1), 2 * i]
                # left of cell
                edges[3 * i + 1, :] = [2 * i, 2 * i + 1]
                # top of cell
                edges[3 * i + 2, :] = [2 * i + 1, 2 * (i + 1) + 1]
        # far right hand edge
        edges[3 * ncells, :] = [2 * ncells, 2 * ncells + 1]
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=4)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def make_1D_grid_Cartesian_non_orthog(self, L, show_grid=False):
        """
        Setup unstructured grid for channel 1 cell wide
        Grid is Cartesian with edge length = self.dx
          but then has random displacement of nodes to create non-orothgonality
        """
        ncells = int(L / self.dx)
        npoints = 2 * ncells + 2
        nedges = 3 * ncells + 1
        points = np.zeros((npoints, 2), np.float64)
        # in future if only 3 edges, node4 will have value -1
        cells = -1 * np.ones((ncells, 4), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        max_displacement = 0.4
        for i in range(ncells + 1):  # set node x,y etc.
            disp = 2 * max_displacement * random.random() - max_displacement
            points[2 * i, 0] = self.dx * i + disp
            disp = 2 * max_displacement * random.random() - max_displacement
            points[2 * i, 1] = 0.0 + disp
            disp = 2 * max_displacement * random.random() - max_displacement
            points[2 * i + 1, 0] = self.dx * i + disp
            disp = 2 * max_displacement * random.random() - max_displacement
            points[2 * i + 1, 1] = self.dx + disp
            if i < ncells:
                cells[i, :] = [2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1, 2 * i]
                # bottom of cell
                edges[3 * i, :] = [2 * (i + 1), 2 * i]
                # left of cell
                edges[3 * i + 1, :] = [2 * i, 2 * i + 1]
                # top of cell
                edges[3 * i + 2, :] = [2 * i + 1, 2 * (i + 1) + 1]
        # far right hand edge
        edges[3 * ncells, :] = [2 * ncells, 2 * ncells + 1]
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=4)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def make_1D_grid_equilat_tri(self, L, show_grid=False):
        """
        Setup unstructured grid for channel of equilateral triangles
        Edge length = self.dx
        """
        n = int(L / self.dx)
        ncells = 2 * n
        npoints = 2 * n + 2
        nedges = 4 * n + 1
        points = np.zeros((npoints, 2), np.float64)
        # in future if only 3 edges, node4 will have value -1
        cells = -1 * np.ones((ncells, 3), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        for i in range(n + 1):  # set node x,y etc.
            points[2 * i, 0] = self.dx * i
            points[2 * i, 1] = 0.0
            points[2 * i + 1, 0] = self.dx * i + 0.5 * self.dx
            points[2 * i + 1, 1] = self.dx * np.sqrt(3.) / 2.
            if i < n:
                cells[2 * i, :] = [2 * (i + 1), 2 * i + 1, 2 * i]
                cells[2 * i + 1, :] = [2 * (i + 1) + 1, 2 * i + 1, 2 * (i + 1)]
                # bottom of cell
                edges[4 * i, :] = [2 * (i + 1), 2 * i]
                # left of cell
                edges[4 * i + 1, :] = [2 * i, 2 * i + 1]
                # top of cell
                edges[4 * i + 2, :] = [2 * i + 1, 2 * (i + 1) + 1]
                # diag of cell
                edges[4 * i + 3, :] = [2 * (i + 1), 2 * i + 1]
        # far right hand edge
        edges[4 * n, :] = [npoints - 1, npoints - 2]
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=3)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def make_2D_grid_equilat_tri(self, L, show_grid=False):
        """
        Setup unstructured grid for channel of equilateral triangles
        Edge length = self.dx
        Number of triangles across = nwide
        """
        n = int(L / self.dx)
        nwide = 4
        ncells = 2 * nwide * n
        npoints = (n + 1) * (nwide + 1)
        nedges = (3 * nwide + 1) * n + nwide
        points = np.zeros((npoints, 2), np.float64)
        # in future if only 3 edges, node4 will have value -1
        cells = -1 * np.ones((ncells, 3), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        edge_ct = 0
        for i in range(n + 1):  # set node x,y etc.
            for k in range(nwide + 1):
                points[(nwide + 1) * i + k, 0] = self.dx * i + 0.5 * self.dx * float(k)
                points[(nwide + 1) * i + k, 1] = k * self.dx * np.sqrt(3.) / 2.
            if i < n:
                for k in range(nwide):
                    idx = nwide * i * 2 + 2 * k
                    cells[idx, :] = [(nwide + 1) * (i + 1) + k, (nwide + 1) * i + k + 1, (nwide + 1) * i + k]
                    cells[idx + 1, :] = [(nwide + 1) * (i + 1) + k, (nwide + 1) * (i + 1) + k + 1, (nwide + 1) * i + k + 1]
                # Left edges of block
                for k in range(nwide):
                    edges[edge_ct, :] = [(nwide + 1) * i + k, (nwide + 1) * i + k + 1]
                    edge_ct += 1
                # Zigzag edges
                for k in range(nwide):
                    edges[edge_ct, :] = [(nwide + 1) * (i + 1) + k, (nwide + 1) * i + k]
                    edge_ct += 1
                    edges[edge_ct, :] = [(nwide + 1) * (i + 1) + k, (nwide + 1) * i + k + 1]
                    edge_ct += 1
                # Top edge of block
                k = nwide - 1
                edges[edge_ct, :] = [(nwide + 1) * (i + 1) + k + 1, (nwide + 1) * i + k + 1]
                edge_ct += 1
        # far right hand edges
        for k in range(nwide):
            edges[edge_ct, :] = [(nwide + 1) * n + k, (nwide + 1) * n + k + 1]
            edge_ct += 1
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=3)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def import_ras_geometry(self, hdf_fname, twod_area_name, max_cell_faces, show_grid=False):

        h = h5py.File(hdf_fname, 'r')

        cell_center_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells Center Coordinate']
        ncells = len(cell_center_xy)
        ccx = np.array([cell_center_xy[i][0] for i in range(ncells)])
        ccy = np.array([cell_center_xy[i][1] for i in range(ncells)])

        points_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/FacePoints Coordinate']
        npoints = len(points_xy)
        points = np.zeros((npoints, 2), np.float64)
        for n in range(npoints):
            points[n, 0] = points_xy[n][0]
            points[n, 1] = points_xy[n][1]

        edge_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces FacePoint Indexes']
        nedges = len(edge_nodes)
        edges = -1 * np.ones((nedges, 2), dtype=int)
        for j in range(nedges):
            edges[j][0] = edge_nodes[j][0]
            edges[j][1] = edge_nodes[j][1]

        cell_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells FacePoint Indexes']
        for i in range(len(cell_nodes)):
            if cell_nodes[i][2] < 0:  # first ghost cell (which are sorted to end of list)
                break
        ncells = i  # don't count ghost cells
        # ncells = len(cell_nodes)
        cells = -1 * np.ones((ncells, max_cell_faces), dtype=int)
        for i in range(ncells):
            for k in range(max_cell_faces):
                cells[i][k] = cell_nodes[i][k]

        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=max_cell_faces)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def set_grid(self, case, domain_length=100., grid_type='1D_Cartesian', show_grid=False):
        """
        Set up grid for simulation
        """
        self.domain_length = domain_length
        self.grid_type = grid_type
        if 'dam_break' in case:
            if grid_type == '1D_Cartesian':
                self.make_1D_grid_Cartesian(domain_length, show_grid=show_grid)
                # self.make_1D_grid_Cartesian_non_orthog(domain_length, show_grid=show_grid)
            elif grid_type == '1D_tri':
                self.make_1D_grid_equilat_tri(domain_length, show_grid=show_grid)
            elif grid_type == '2D_tri':
                self.make_2D_grid_equilat_tri(domain_length, show_grid=show_grid)
            elif grid_type == 'RAS':
                # self.import_ras_geometry(r'J:\work_2018\RAS-2D\grid_generation\grid_gen.g03.hdf',
                #                          'a', 7, show_grid=show_grid)  # dx=5, first cut line
                # self.import_ras_geometry(r'J:\work_2018\RAS-2D\grid_generation\grid_gen.g05.hdf',
                #                          'a', 4, show_grid=show_grid)  # dx=5, no cut line
                self.import_ras_geometry(r'J:\work_2018\RAS-2D\grid_generation\grid_gen.g06.hdf',
                                         'a', 8, show_grid=show_grid)  # dx=5, severe cut line
        elif 'bump' in case:
            if grid_type == '1D_Cartesian':
                self.make_1D_grid_Cartesian(domain_length, show_grid=show_grid)

        # compute topology using unstructured_grid class methods
        self.set_topology()
        print('ncells', self.ncells)

        return

    def set_initial_conditions(self, case, us_eta=0., ds_eta=0., us_q=0.):
        """
        Set up initial conditions for simulation
        """
        # allocates the IC arrays
        super(SWAMPpy,self).set_initial_conditions()
        # set initial conditions

        if 'bump' in case:
            self.use_contract_factor = True
        else:
            self.use_contract_factor = False

        if case == 'dam_break':
            self.ds_eta = ds_eta
            self.ic_ei[:] = ds_eta  # downstream wse
            ileft = np.where(self.grd.cells['_center'][:, 0] < self.domain_length / 2.)[0]
            # ileft = np.where(self.grd.cells['_center'][:, 0] > domain_length / 2.)[0]
            self.ic_ei[ileft] = us_eta

        # set initial conditions - bump
        elif 'bump' in case:
            self.ds_eta = ds_eta
            self.qinflow = us_q
            self.ic_ei[:] = ds_eta
            self.ds_i = np.where(self.grd.cells['_center'][:, 0] > (self.domain_length - self.dx))[0]
            self.us_i = np.where(self.grd.cells['_center'][:, 0] < (0. + self.dx))[0]
            self.us_j = np.where(self.exy[:][0] < (0. + self.dx / 4.))[0]
            if case == 'bump':
                for i in range(self.ncells):
                    xtmp = self.grd.cells['_center'][i, 0]
                    if xtmp >= 8. and xtmp <= 12.:
                        self.ic_zi[i] = -(0.2 - 0.05 * (xtmp - 10.) ** 2)
            elif case == 'bump2':
                self.ic_zi[:] = 1.0
                for i in range(self.ncells):
                    xtmp = self.grd.cells['_center'][i, 0]
                    if xtmp >= 45. and xtmp <= 55.:
                        self.ic_zi[i] = 0.0
                    elif xtmp >= 40. and xtmp < 45.:
                        self.ic_zi[i] = (45. - xtmp) * 0.2
                    elif xtmp >= 55. and xtmp < 60.:
                        self.ic_zi[i] = (xtmp - 55.) * 0.2
            elif case=='bump_gauss':
                xtmp=self.grd.cells['_center'][:,0]
                self.ic_zi[:] = -1.5*np.exp(-((xtmp-500.0)/50.)**2)
        return

    def step_output(self,n,ei,**kwargs):
        """
        n: timestep
        ei: freesurface
        """
        if n % 100 == 0:
            # Comparison plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.grd.cells['_center'][:, 0], ei, 'r.', ms=2, label='model')
            ax.plot(self.grd.cells['_center'][:, 0], -self.ic_zi, '-k', label='bed')
            ax.set_ylabel('Water Surface Elevation, m')
            ax.legend(loc='upper right')
            plt.savefig(str(n) + '.png', bbox_inches='tight')
            plt.close('all')

def get_dry_dam_break_analytical_soln(domain_length, upstream_height, tend):
    """
    Return x, h, u for analytical dry dam break solution
    Solution from Delestre et al, 2012 4.1.2
    dx here just applies to resolution of analytical solution
    """
    dx = 1.0
    x0 = domain_length / 2.0
    t = tend
    sgh = np.sqrt(g * upstream_height)
    xA = x0 - t * sgh
    xB = x0 + 2.0 * t * sgh
    imx = np.int(domain_length / dx)
    xj = np.asarray([dx * float(i) for i in range(imx + 1)])
    xi = 0.5 * (xj[:-1] + xj[1:])
    han = np.zeros(imx, np.float64)
    uan = np.zeros_like(han)

    for i in range(imx):
        if xi[i] < xA:
            han[i] = upstream_height
            uan[i] = 0.0
        elif xi[i] > xB:
            han[i] = 0.0
            uan[i] = 0.0
        else:
            xt = xi[i] - x0
            han[i] = (4.0 / (9.0 * g)) * (sgh - xt / (2.0 * t)) ** 2
            uan[i] = (2.0 / 3.0) * (xt / t + sgh)

    return (xi, han, uan)


def get_bump_observed():
    """
    Read in observed data from lab experiment
    """
    fname = r'J:\work_2018\RAS-2D\bump_obs_results.csv'
    x = []
    y = []
    count = 0
    if not os.path.exists(fname):
        return None
    for line in open(fname, 'r'):
        if not line:
            break
        count += 1
        if count < 2:  # header line
            continue
        sline = line.split(',')
        x.append(float(sline[0]))
        y.append(float(sline[1]))
    return (np.array(x), np.array(y))


def get_bump2_analytic(domain_length):
    """
    Get analytic solution for flow over a bc weir
    """
    dx = 1.0
    imx = np.int(domain_length / dx)
    xan = np.arange(0., domain_length, imx) + dx / 2.
    han = np.zeros(imx, np.float64)
    return (xan, han)


# to do
# - modularize with options
# - embed in loops to test sensitivity
# - improve traceback step
# - implement wet-wet case
# - implement theta method
# - add slope limiter
# - implement conservative form


if __name__ == '__main__':

    case = 'bump_gauss'  # dry_dam_break bump (later, wet_dam_break)

    if case == 'bump':
        dx = 0.05
        dt = 0.05
        tend = 100.
        domain_length = 20.
        grid_type = '1D_Cartesian'
        # upstream_flow = 0.18  # m2/s
        upstream_flow = 0.08 # m2/s -- RH: avoid hydraulic jump
        downstream_eta = 0.33
    elif case == 'bump2':
        dx = 0.25
        dt = 0.025
        tend = 200.
        domain_length = 100.
        grid_type = '1D_Cartesian'
        upstream_flow = 1.0  # m2/s
        downstream_eta = 0.0
    elif case=='bump_gauss':
        dx = 5.0
        dt = 0.5
        tend = 4*3600.
        domain_length = 1000.
        grid_type = '1D_Cartesian'
        upstream_flow = 200.0/50.0  # m2/s
        downstream_eta = 5.0

    elif case == 'dam_break':
        dt = 0.05  # 0.01 for dx=1 Cartesian grid, 0.005 for triangle grid, 0.025 for RAS
        dx = 5.0  # 5 for ras grids, 1 otherwise
        tend = 10.
        domain_length = 1200.
        grid_type = '1D_Cartesian'  # 1D_Cartesian 1D_tri 2D_tri RAS
        upstream_height = 10.
        downstream_height = 0.

    swampy = SWAMPpy(dt=dt, dx=dx)
    swampy.set_grid(case, domain_length=domain_length, grid_type=grid_type, show_grid=False)

    if 'bump' in case:
        swampy.set_initial_conditions(case, ds_eta=downstream_eta, us_q=upstream_flow)
    elif case == 'dam_break':
        swampy.set_initial_conditions(case, us_eta=upstream_height, ds_eta=downstream_height)

    # hi: 400x = ei + zi
    # uj: velocity on all edges
    # tvol?
    # ei: 400: 
    (hi, uj, tvol, ei) = swampy.run(case, tend)  # final water surface elevation, velocity
    ui = swampy.get_center_vel(uj)
    #     hjstar = swampy.calc_hjstar(ei, swampy.ic_zi, uj)
    #     ui = swampy.get_center_vel_hweight(uj, hjstar, hi)
    (xe, uxe) = swampy.get_edge_x_vel(uj)
    (xc, uxc) = swampy.get_xsect_avg_val(ui[:, 0])
    (xc, ec) = swampy.get_xsect_avg_val(ei)

    if case == 'dam_break':

        # dry dam break analytical solution
        if swampy.ds_eta > 0.001:
            (xan, han, uan) = get_dry_dam_break_analytical_soln(domain_length, upstream_height, tend)  # implement wet dam break solution
        else:
            (xan, han, uan) = get_dry_dam_break_analytical_soln(domain_length, upstream_height, tend)

        # Analytical comparison plot
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(xan, han, '-b', label='analytical', linewidth=1.5)
        # ax1.plot(swampy.grd.cells['_center'][:, 0], ei, 'r.', ms=2, label='model')  # , linewidth=1.5)
        ax1.plot(xc, ec, 'r.', ms=2, label='model')  # , linewidth=1.5)
        ax1.set_ylabel('Depth, m')
        ax1.legend(loc='upper right')
        ax2 = fig.add_subplot(212)
        ax2.plot(xan, uan, '-b', label='analytical', linewidth=1.5)
        ax2.plot(xc, uxc, 'r.', ms=2, label='model')
        # ax2.plot(xe, uxe, 'r.', ms=2, label='model, edge velocities')  # , linewidth=1.5)
        # ax2.plot(swampy.grd.cells['_center'][:, 0], ui[:, 0], 'g.', ms=2, label='model, center velocities')  # , linewidth=1.5)
        ax2.set_ylabel('Current Velocity, m/s')
        ax2.set_xlabel('Distance, m')
        ax2.legend(loc='upper left')
        plt.show()

        # Scatter plot showing velocity as a function of edge length
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xan, uan, '-b', label='analytical', linewidth=1.5)
        ax.set_ylabel('Current Velocity, m/s')
        ax.set_xlabel('Distance, m')
        sc = ax.scatter(xe, uxe, s=10, c=swampy.len[swampy.intern], edgecolors='none', cmap=plt.get_cmap('jet'))
        sc.set_label('model')
        cb = plt.colorbar(sc)
        cb.set_label('Edge length (m)')
        plt.legend()
        plt.show()

        if '1D' not in swampy.grid_type:
            # Contour plot of water levels
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xy = swampy.get_grid_poly_collection()
            collection = PolyCollection(xy, cmap=plt.get_cmap('jet'))  # 'Blues' 'jet' 'ocean' mcm
            collection.set_array(ei)
            collection.set_linewidths(0.2)
            collection.set_edgecolors('k')
            ax.add_collection(collection)
            ax.set_xlim([0, domain_length])
            if 'tri' in swampy.grid_type:
                ax.set_ylim([0, 5])
            else:
                ax.set_ylim([0, 100])
                # ax.set_xlim([400, 900])
            ax.set_aspect('equal')

            # Contour plot of water levels and velocity vectors
            fig = plt.figure()
            ax = fig.add_subplot(111)
            collection = PolyCollection(xy, cmap=plt.get_cmap('jet'))  # 'Blues' 'jet' 'ocean' mcm
            collection.set_array(ei)
            collection.set_linewidths(0.2)
            collection.set_edgecolors('k')
            ax.add_collection(collection)
            ax.set_xlim([0, domain_length])
            if 'tri' in swampy.grid_type:
                ax.set_ylim([0, 5])
            else:
                ax.set_ylim([0, 100])
            ax.set_xlim([400, 900])
            ax.quiver(swampy.grd.cells['_center'][:, 0], swampy.grd.cells['_center'][:, 1], ui[:, 0], ui[:, 1])
            ax.set_aspect('equal')

        # Volume conservation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tvol, '-b.')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Total domain volume')

    elif 'bump' in case:
        if case == 'bump':
            try:
                (xan, han) = get_bump_observed()
            except TypeError:
                print("Observed bump data unavailable")
                xan=han=None
            leg_str = 'observed'
        elif case == 'bump2':
            (xan, han) = get_bump2_analytic(domain_length)
            leg_str = 'analytic'
        # Comparison plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if case == 'bump' and xan is not None:
            ax.plot(xan, han, '-b', label=leg_str, linewidth=1.5, zorder=5)
        ax.plot(swampy.grd.cells['_center'][:, 0], ei, 'r.', ms=4, label='model', zorder=5)  # , linewidth=1.5)
        ax.fill(swampy.grd.cells['_center'][:, 0], -swampy.ic_zi, '0.7', label='bed', zorder=5)
        ax.plot(swampy.grd.cells['_center'][:, 0], -swampy.ic_zi, 'k-', zorder=5)
        bernoulli=ui[:,0]**2/(2*9.8) + ei
        ax.plot(swampy.grd.cells['_center'][:, 0], bernoulli, 'b.', ms=4, label='bernoulli model', zorder=5)
        ax.grid(color='0.8', linestyle='-', zorder=3)
        # ax.plot(swampy.grd.cells['_center'][:-1, 0], ei[:-1] + uxe * uxe / (2.*g), '--k', label='E head')
        # ax.plot(swampy.grd.cells['_center'][:-1, 0], hi[:-1] * uxe, ':k', label='Momentum')
        # ax.plot(swampy.grd.cells['_center'][:, 0], ei + ui[:, 0] * ui[:, 0] / (2.*g), '--k', label='E head')
        # ax.plot(swampy.grd.cells['_center'][:, 0], hi * ui[:, 0], ':k', label='Momentum')
        ax.set_ylabel('Water Surface Elevation, m')
        ax.legend(loc='center left')

    plt.show()
    print('Complete!')
