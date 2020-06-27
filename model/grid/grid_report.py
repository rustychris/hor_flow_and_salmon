g=unstructured_grid.UnstructuredGrid.from_ugrid('junction-grid-202-bathy.nc')

##

g.report_orthogonality()

##

# Do any cell centers map to multiple cells?
overlaps=[]
cc=g.cells_center()
for c in range(g.Ncells()):
    if g.select_cells_nearest(cc[c])!=c:
        overlaps.append(c)

if overlaps:
    print("%d cells may overlap others"%len(overlaps))
    for c in overlaps:
        print("  %d (center %.2f,%.2f)"%(c,cc[c,0],cc[c,1]))

##
