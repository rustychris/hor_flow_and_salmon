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
from stompy.grid import unstructured_grid as ugrid

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

# global variables
g = 9.8  # gravity
dc_min = 1.0
max_sides = 8
dzmin = 0.0001
flow_contract_threshold = 100000.01  # 0.01


class SWAMPpy(object):
    def __init__(self,
                 dt=0.025,
                 dx=1.0):
        self.dt = dt
        self.dx = dx

    def center_vel_perot(self, i, cell_uj):
        """
        Cell center velocity calculated with Perot dual grid interpolation
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

    def center_vel_perot_hweight(self, i, cell_uj, cell_hj, cell_hi):
        """
        Cell center velocity calculated with Perot dual grid interpolation
        """
        acu = np.zeros(2, np.float64)
        uside = np.zeros_like(acu)
        ucen = np.zeros_like(acu)
        nsides = len(cell_uj)
        if cell_hi > dzmin:
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                for dim in range(2):  # x and y
                    uside[dim] = cell_uj[l] * self.en[j, dim]  # u vel component of normal vel
                    acu[dim] += uside[dim] * self.len[j] * self.dist[i, l] * cell_hj[l]
            for dim in range(2):  # x and y
                ucen[dim] = acu[dim] / (self.grd.cells['_area'][i] * cell_hi)
        else:
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                for dim in range(2):  # x and y
                    uside[dim] = cell_uj[l] * self.en[j, dim]  # u vel component of normal vel
                    acu[dim] += uside[dim] * self.len[j] * self.dist[i, l]
            for dim in range(2):  # x and y
                ucen[dim] = acu[dim] / self.grd.cells['_area'][i]

        return ucen

    def get_center_vel_hweight(self, uj, hj, hi):
        """
        Get center velocities for every cell in the grid
        """
        ui = np.zeros((self.ncells, 2), np.float64)
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            cell_uj = np.zeros(nsides, np.float64)
            cell_hj = np.zeros(nsides, np.float64)
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                cell_uj[l] = uj[j]
                cell_hj[l] = hj[j]
            cell_hi = hi[i]
            ui[i] = self.center_vel_perot_hweight(i, cell_uj, cell_hj, cell_hi)

        return ui

    def get_center_vel(self, uj):
        """
        Get center velocities for every cell in the grid
        """
        ui = np.zeros((self.ncells, 2), np.float64)
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            cell_uj = np.zeros(nsides, np.float64)
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                cell_uj[l] = uj[j]
            ui[i] = self.center_vel_perot(i, cell_uj)

        return ui

    def get_fu_no_adv(self, uj):
        """
        Return fu term with no advection
        """
        fu = np.zeros_like(uj)
        for j in self.intern:  # loop over internal cells
            fu[j] = uj[j]  # no advection

        return fu

    def get_fu_orig(self, uj):
        """
        Return fu term with original advection method
        """
        fu = np.zeros_like(uj)
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
            fu[j] = uj[j] * (1. - 2.*self.dt * np.sign(uj[j]) * (uj[j] - ui_norm) / self.dc[j])

        return fu

    def get_fu_Perot(self, uj, alpha, sil, hi, hjstar, hjbar, hjtilde, use_contract_factor=False):
        """
        Return fu term with Perot method
        """
        fu = np.zeros_like(uj)
        ui = self.get_center_vel(uj)
        # ui = self.get_center_vel_hweight(uj, hjstar, hi)
        for j in self.intern:
            fu[j] = uj[j]
            ii = self.grd.edges[j]['cells']
            iL = ii[0]
            iR = ii[1]
            cv_vol = alpha[j, 0] * self.grd.cells['_area'][iL] * hi[iL] + alpha[j, 1] * self.grd.cells['_area'][iR] * hi[iR]
            cv_area = self.grd.cells['_area'][iL] + self.grd.cells['_area'][iR]
            if hjbar[j] < dzmin:  # no advection term if no water at a cell edge
                # if cv_vol / cv_area < dzmin:  # no advection term if no water at a cell edge
                continue
            # calculate contraction factor
            factor = 1.
            if use_contract_factor and (hi[iR] > dzmin and hi[iL] > dzmin):
                if uj[j] > 0.:
                    flow_contract_factor = (hi[iL] - hi[iR]) / self.dc[j]
                else:
                    flow_contract_factor = (hi[iR] - hi[iL]) / self.dc[j]
                if flow_contract_factor > flow_contract_threshold:
                    factor = hjbar[j] * (hi[iL] + hi[iR]) / (2.*hi[iL] * hi[iR])

            # left cell
            sum1 = 0.
            nsides = self.ncsides[iL]
            for l in range(nsides):
                k = self.grd.cells[iL]['edges'][l]
                Q = sil[iL, l] * self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                i2 = iitmp[np.nonzero(iitmp - iL)[0][0]]  # get neighbor
                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])
            fu[j] -= factor * self.dt * alpha[j, 0] * sum1 / (self.grd.cells['_area'][iL] * hjbar[j])
            # fu[j] -= self.dt * alpha[j, 0] * sum1 / cv_vol
            # right cell
            sum1 = 0.
            nsides = self.ncsides[iR]
            for l in range(nsides):
                k = self.grd.cells[iR]['edges'][l]
                Q = sil[iR, l] * self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                i2 = iitmp[np.nonzero(iitmp - iR)[0][0]]  # get neighbor
                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])
            fu[j] -= factor * self.dt * alpha[j, 1] * sum1 / (self.grd.cells['_area'][iR] * hjbar[j])
            # fu[j] -= self.dt * alpha[j, 1] * sum1 / cv_vol

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
        Original method of calculating h at faces
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
            hjstar[j] = min(zi[ii[0]], zi[ii[1]]) + e_up

        return hjstar

    def calc_hjbar(self, ei, zi):
        """
        Calculate depth at a face
        """
        hjbar = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            # esg - could use hi directly instead of recalculating
            hL = ei[ii[0]] + zi[ii[0]]
            hR = ei[ii[1]] + zi[ii[1]]
            hjbar[j] = self.alpha[j][0] * hL + self.alpha[j][1] * hR

        return hjbar

    def calc_hjtilde(self, ei, zi):
        """
        Calculate depth at a face
        """
        hjtilde = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            eavg = 0.5 * (ei[ii[0]] + ei[ii[1]])
            bL = zi[ii[0]]
            bR = zi[ii[1]]
            hjtilde[j] = eavg + self.alpha[j][0] * bL + self.alpha[j][1] * bR

        return hjtilde

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

    def set_topology(self):
        """
        Use functions of unstructured grid class for remaining topology
        """
        self.nedges = len(self.grd.edges)
        self.ncells = len(self.grd.cells)
        self.nnodes = len(self.grd.nodes)
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
        for j in self.intern:
            ii = self.grd.edges[j]['cells']  # 2 cells
            xy = self.grd.cells[ii]['_center']  # 2x2 array
            dc[j] = euclidean(xy[0], xy[1])  # from scipy
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
                dist[i, l] = euclidean(side_xy, cen_xy)  # from scipy

        return dist

    def get_alphas_Perot(self):
        """
        Return Perot weighting coefficients
        Coefficients for external edges are set to zero (should not be used)
        """
        alpha = np.zeros((self.nedges, 2), np.float64)
        for j in self.intern:
            side_xy = self.exy[j]
            ii = self.grd.edges[j]['cells']
            cen_xyL = self.grd.cells['_center'][ii[0]]
            cen_xyR = self.grd.cells['_center'][ii[1]]
            alpha[j, 0] = euclidean(side_xy, cen_xyL) / self.dc[j]
            alpha[j, 1] = euclidean(side_xy, cen_xyR) / self.dc[j]
        self.alpha = alpha

        return alpha

    def get_alphas_Wenneker(self):
        """
        Return Wenekker weighting coefficients
        Coefficients for external edges are set to zero (should not be used)
        """
        alpha = np.zeros((self.nedges, 2), np.float64)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']
            Al = self.grd.cells['_area'][ii[0]]
            Ar = self.grd.cells['_area'][ii[1]]
            alpha[j, 0] = Al / (Al + Ar)
            alpha[j, 1] = Ar / (Al + Ar)
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
        # set initial conditions
        self.ic_ei = np.zeros(self.ncells, np.float64)
        self.ic_zi = np.zeros_like(self.ic_ei)

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

    def run(self, case, tend):
        """
        Run simulation
        """
        dt = self.dt
        self.tend = tend
        nsteps = np.int(np.round(tend / dt))

        # precompute constants
        gdt2 = g * dt * dt
        dc = self.get_cell_center_spacings()
        self.dist = self.edge_to_cen_dist()
        alpha = self.get_alphas_Perot()

        sil = self.get_sign_array()
        # lij = self.get_side_num_of_cell()

        # edge length divided by center spacing
        len_dc_ratio = np.divide(self.len, dc)

        # cell center values
        ncells = self.ncells
        hi = np.zeros(ncells, np.float64)  # total depth
        zi = np.zeros_like(hi)  # bed elevation, measured positive down
        ei = np.zeros_like(hi)  # water surface elevation
        # edge values
        uj = np.zeros(self.nedges, np.float64)  # normal velocity at side
        qj = np.zeros(self.nedges, np.float64)  # normal velocity*h at side
        # Matrix
        A = np.zeros((ncells, ncells), np.float64)
        b = np.zeros(ncells, np.float64)
        # short vars for grid
        area = self.grd.cells['_area']

        # set initial conditions
        ei[:] = self.ic_ei[:]
        zi[:] = self.ic_zi[:]

        hi = zi + ei  # update total depth
        hjstar = self.calc_hjstar(ei, zi, uj)
        hjbar = self.calc_hjbar(ei, zi)
        hjtilde = self.calc_hjtilde(ei, zi)
        if 'bump' in case:
            use_contract_factor = True
        else:
            use_contract_factor = False

        # change code to vector operations at some point
        # time stepping loop
        tvol = np.zeros(nsteps)
        for n in range(nsteps):
            print('step', n + 1, 'of', nsteps)

            # calculate advection term
            # fu = self.get_fu_no_adv(uj)
            # fu = self.get_fu_orig(uj)
            fu = self.get_fu_Perot(uj, alpha, sil, hi, hjstar, hjbar, hjtilde, use_contract_factor)

            # set matrix coefficients
            # first put one on diagonal, and explicit stuff on rhs (b)
            A[:, :] = 0.
            b[:] = 0.
            for i in range(ncells):
                A[i, i] = 1.0
                s_len_h_u_sum = 0.0
                for l in range(self.ncsides[i]):
                    j = self.grd.cells[i]['edges'][l]
                    if self.grd.edges[j]['mark'] == 0:  # if internal
                        s_len_h_u_sum += sil[i, l] * self.len[j] * hjstar[j] * fu[j]
                        if hjbar[j] > dzmin:
                            hterm = hjstar[j] * hjtilde[j] / hjbar[j]
                        else:
                            hterm = hjstar[j]
                        coeff = (gdt2 / area[i]) * len_dc_ratio[j] * hterm
                        A[i, i] += coeff
                        ii = self.grd.edges[j]['cells']  # 2 cells
                        i2 = ii[np.nonzero(ii - i)[0][0]]  # i2 = index of neighbor
                        A[i, i2] = -coeff
                b[i] = ei[i] - (dt / area[i]) * s_len_h_u_sum
            if 'bump' in case:
                A[self.ds_i, :] = 0.
                A[self.ds_i, self.ds_i] = 1.
                b[self.ds_i] = self.ds_eta
                b[self.us_i] = b[self.us_i] + (dt / area[self.us_i]) * self.len[self.us_j] * self.qinflow

            # invert matrix
            start = time.time()
            ei, success = sparse.linalg.cg(A, b, tol=1.e-12)  # Decrease tolerance for faster speed
            end = time.time()
            print('matrix solve took', '{0:0.4f}'.format(end - start), 'sec')
            if success != 0:
                raise RuntimeError('Error in convergence of conj grad solver')

            hi = zi + ei
            # substitute back into u solution
            for j in self.intern:  # loop over internal cells
                ii = self.grd.edges[j]['cells']  # 2 cells
                if hjbar[j] > dzmin:
                    uj[j] = fu[j] - (hjtilde[j] / hjbar[j]) * (g * dt / dc[j]) * (ei[ii[1]] - ei[ii[0]])
                else:
                    uj[j] = fu[j] - (g * dt / dc[j]) * (ei[ii[1]] - ei[ii[0]])

            # Update elevations
            hi = ei + zi
            hjstar = self.calc_hjstar(ei, zi, uj)
            hjbar = self.calc_hjbar(ei, zi)
            hjtilde = self.calc_hjtilde(ei, zi)

            # conservation properties
            tvol[n] = np.sum(hi * self.grd.cells['_area'][:])

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


## Overlay suntans output:
import xarray as xr

ds_adv=xr.open_dataset('../suntans/runs/straight_03/Estuary_SUNTANS.nc.nc.0')
ds_noa=xr.open_dataset('../suntans/runs/straight_04/Estuary_SUNTANS.nc.nc.0')

##

#plt.figure(1).clf()
#fig,ax=plt.subplots(num=1)
offset=5.0

for ds,ls,label in [(ds_adv,'-','sun_adv'),
                    (ds_noa,'--','sun_lin')]:
    x=ds.xv.values
    ord=np.argsort(x)

    # Eta
    ax.plot( x[ord], offset+ds.eta.isel(time=-1).values[ord],color='r',ls=ls,label=label+' eta')

    # bed
    ax.plot( x[ord], offset-ds.dv.values[ord], color='k',ls=ls, label=label+'bed')

    # bernoulli
    eta=ds.eta.isel(time=-1).values
    u=ds.uc.isel(time=-1,Nk=0).values
    bern=eta + u**2/(2*9.8)
    ax.plot( x[ord], offset+bern,color='b',ls=ls,label=label+' bern')
