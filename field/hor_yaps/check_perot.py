
def add_U_perot(snap):
    cc=g.cells_center()
    ec=g.edges_center()
    normals=g.edges_normals()
    
    e2c=g.edge_to_cells()
    
    Uclayer=np.zeros( (g.Ncells(),snap.dims['nMesh2_layer_3d'],2), np.float64)
    Qlayer=snap.h_flow_avg.values
    Vlayer=snap.Mesh2_face_water_volume.values
    
    dist_edge_face=np.nan*np.zeros( (g.Ncells(),g.max_sides), np.float64)
        
    for c in np.arange(g.Ncells()):
        js=g.cell_to_edges(c)
        for nf,j in enumerate(js):
            # normal points from cell 0 to cell 1
            if e2c[j,0]==c: # normal points away from c
                csgn=1
            else:
                csgn=-1
            dist_edge_face[c,nf]=np.dot( (ec[j]-cc[c]), normals[j] ) * csgn
            # Uc ~ m3/s * m
            Uclayer[c,:,:] += Qlayer[j,:,None]*normals[j,None,:]*dist_edge_face[c,nf]
            
    Uclayer /= np.maximum(Vlayer,0.01)[:,:,None]
    snap['uc']=('nMesh2_layer_3d','nMesh2_face'),Uclayer[:,:,0].transpose()
    snap['vc']=('nMesh2_layer_3d','nMesh2_face'),Uclayer[:,:,1].transpose() 
