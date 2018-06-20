
all: derived/grid-edges.shp derived/grid_ugrid.nc derived/grid_net.nc

# -D: don't set timestamp (which would confuse make down the road)
# -d derived: unzip there.
derived/junction_29_w_depth_2007.grd: ../../untrim/ed-steady/junction_29_w_depth_2007.zip
	unzip -d derived $<
	touch $@ # otherwise timestamps are messed

derived/grid_ugrid.nc: derived/junction_29_w_depth_2007.grd
	python -m stompy.grid.cli -i untrim:derived/junction_29_w_depth_2007.grd -co ugrid:$@

derived/grid_bathy_ugrid.nc: derived/grid_ugrid.nc
	python add_bathy.py $< $@

derived/grid_net.nc: derived/grid_bathy_ugrid.nc
		python -m stompy.grid.cli -i ugrid:$< -co dfm:$@

derived/grid-edges.shp: derived/grid_ugrid.nc
	python -m stompy.grid.cli -i ugrid:$< -co edge_shp:$@

