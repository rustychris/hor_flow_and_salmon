library(yaps)

# load tag data somewhat pre-aligned.

#pings_nc<-"../circulation/pings-2018-03-17T16:08:23.763998000_2018-03-17T18:05:33.221453000.nc"

#library(chron)
#library(lattice)
#library(ncdf4)

#pings_in<-ncdf4::nc_open(pings_nc)

# careful parsing missing sync_tag here
# also, any sync_tag that is not received by at least 3 rxs (self+2)
# at least once, causes issues. Those have been filtered out upstream.
hydros<-data.table::fread("hydros.csv",fill=TRUE,na.strings=c(""))
detections<-data.table::fread("detections.csv",fill=TRUE)

beacon2018<-c()
beacon2018$hydros <- hydros
beacon2018$detections <- detections

inp_sync <- getInpSync(sync_dat=beacon2018, max_epo_diff = 10,
                       min_hydros = 2,
                       time_keeper_idx = 2,
                       fixed_hydros_idx = 1:13,
                       n_offset_day = 4,
                       n_ss_day = 2)

# This seems to be working -- gets quite small residuals.
sync_model <- getSyncModel(inp_sync,silent=TRUE)

plotSyncModelResids(sync_model,by="overall")
plotSyncModelResids(sync_model,by="sync_tag")
plotSyncModelResids(sync_model,by="hydro")
plotSyncModelCheck(sync_model,by="sync_bin_sync")
plotSyncModelCheck(sync_model,by="sync_bin_hydro")
plotSyncModelCheck(sync_model,by="sync_tag")
plotSyncModelCheck(sync_model,by="hydro")

#==
# The example process --
fn <- system.file("extdata","VUE_Export_ssu1.csv",package="yaps")
vue <- data.table::fread(fn,fill=TRUE)
detections <- prepDetections(raw_dat=vue,type="vemco_vue")

# where does ssu1 come from? they sneak it in when loading
# yaps. 
# ssu1 has $hydros => serial, x y, z, sync_tag, idx
#   sync_tag is the beacon tag, and idx is just 1:19
# $detections => ts, tag, epo, frac, serial
#   i.e. detections of beacon tags.
# $gps => I think this is the test tag trajectory.

inp_sync <- getInpSync(sync_dat=ssu1, max_epo_diff = 120,
                       min_hydros = 2,
                       time_keeper_idx = 5,
                       fixed_hydros_idx = c(2:3,6,8,11,13:17),
                       n_offset_day = 2,
                       n_ss_day = 2)

sync_model <- getSyncModel(inp_sync,silent=TRUE)

plotSyncModelResids(sync_model,by="overall")
# and some others...

# Maybe this is a good point at which to try bringing in my data?
  