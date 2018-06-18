
all: derived/grid-edges.shp derived/grid_ugrid.nc derived/grid_net.nc

derived/grid_ugrid.nc: junction_29_w_depth_2007.grd
		python -m stompy.grid.cli -i untrim:junction_29_w_depth_2007.grd -co ugrid:$@

derived/grid_net.nc: derived/grid_ugrid.nc
		python -m stompy.grid.cli -i ugrid:$< -co dfm:$@

derived/grid-edges.shp: derived/grid_ugrid.nc
	python -m stompy.grid.cli -i ugrid:$< -o edge_shp:$@

