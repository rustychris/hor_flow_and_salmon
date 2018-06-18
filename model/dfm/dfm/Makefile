DFM_HOME=$(HOME)/src/dfm/r52184-dbg/bin

FORCING_SHP=forcing-v00.shp

NPROCS=4

MPIHOME?=$(DFM_HOME)
MPIEXEC=$(MPIHOME)/mpiexec
DFLOWFM=$(DFM_HOME)/dflowfm

DYLD_LIBRARY_PATH=$(DFM_HOME)/../lib
LD_LIBRARY_PATH=$(DFM_HOME)/../lib
export DYLD_LIBRARY_PATH LD_LIBRARY_PATH

PYTHON=python

DEFAULT_RUN=basic

RUN?=$(DEFAULT_RUN)

RUN_DIR=runs/$(RUN)

MDU=$(RUN_DIR)/flowfm.mdu

GRID=$(RUN_DIR)/grid_net.nc

all: plis partition

plis: runs/$(RUN)/$(FORCING_SHP).plis

ifneq ($(NPROCS), 1)
partition: runs/$(RUN)/grid-partition-$(NPROCS) runs/$(RUN)/mdu-partition-$(NPROCS)
else
partition:
	@echo No partitioning for nprocs=$(NPROCS) 
endif


$(RUN_DIR):
	[ -d $(RUN_DIR) ] || mkdir $(RUN_DIR)

runs/$(RUN)/$(FORCING_SHP).plis: $(RUN_DIR) gis/$(FORCING_SHP)
	$(PYTHON) generate_plis.py gis/$(FORCING_SHP) $(RUN_DIR)
	touch $@

runs/$(RUN)/grid-partition-$(NPROCS): $(GRID)
	echo $(DYLD_LIBRARY_PATH)
	$(MPIEXEC) -n $(NPROCS) $(DFLOWFM) --partition:ndomains=$(NPROCS) $(GRID)
	touch $@

runs/$(RUN)/mdu-partition-$(NPROCS): $(MDU)
	cd $(RUN_DIR) && $(DFM_HOME)/generate_parallel_mdu.sh $(shell basename $(MDU)) $(NPROCS) 6
	touch $@


