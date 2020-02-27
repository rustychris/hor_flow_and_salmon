library(yaps)

# load tag data somewhat pre-aligned.
# careful parsing missing sync_tag here
# also, any sync_tag that is not received by at least 3 rxs (self+2)
# at least once, causes issues. Those have been filtered out upstream.
hydros<-data.table::fread("hydros.csv",fill=TRUE,na.strings=c(""))
all_detections<-data.table::fread("all_detections.csv",fill=TRUE)
#beacon_detections<-data.table::fread("beacon_detections.csv",fill=TRUE)

beacon2018<-c()
beacon2018$hydros <- hydros
# for getting the syncmodel:
beacon2018$detections <- all_detections
# beacon_detections

# seems that hydro 11 has very little data, also hydro 1.
# those are AM3 (11) and SM9 (1)
# we're getting a bad sync on station 11, maybe, and that's causing 
# pings to get discarded because of rbi_min.
# including 11 (AM3) fails.
# omitting 1, 11, and the potentially swapped 3,4 at least completes.
# n_offset_day 4=>1 (in hopes of getting something for 11 and 1.)
# 1 isn't allowed. how about 2? fails to complete.
# go back and pull a longer chunk of data.
# that failed to complete.  Maybe too many fixed hydros?
# with c(5:10) -- it's taking its sweet time.  but it does complete.
# But the end result is no better.  it still gets a bunch of bad timestamps from 11.
# so now the input files above have 11 prefiltered out.
inp_sync <- getInpSync(sync_dat=beacon2018, max_epo_diff = 10,
                       min_hydros = 2,
                       time_keeper_idx = 2,
                       fixed_hydros_idx = c(1:12), # 1:13 fails!
                       n_offset_day = 4,
                       n_ss_day = 2)

# This seems to be working -- a few outliers, but generally O(0.1m) residuals.
sync_model <- getSyncModel(inp_sync,silent=FALSE)

# seem to be having some trouble while bringing in the rest of the detections.
# running all of the above code, but in a non-fresh environment...
# no different running in fresh environment.
# that was with including 2-rx pings for beacons.  How about going back to forcing
# beacon pings to be received by 3 (in addition to skipping )
# is it b/c I set too many to fixed?
# hmm - 11 and 1 don't even get times for the latter 2 offset intervals.

plotSyncModelResids(sync_model,by="overall")
plotSyncModelResids(sync_model,by="sync_tag")
plotSyncModelResids(sync_model,by="hydro")
plotSyncModelCheck(sync_model,by="sync_bin_sync")
plotSyncModelCheck(sync_model,by="sync_bin_hydro")
plotSyncModelCheck(sync_model,by="sync_tag")
plotSyncModelCheck(sync_model,by="hydro")

detections_synced <- applySync(toa=all_detections, hydros=hydros, sync_model)

hydros_yaps <- data.table::data.table(sync_model$pl$TRUE_H)
colnames(hydros_yaps) <- c('hx','hy','hz')

# Choose a tag:
focal_tag <- '7ADB'
rbi_min <- 4 # min burst interval
rbi_max <- 7 # max burst interval

synced_dat <- detections_synced[tag==focal_tag]

# why rows with all nan?
# 
toa <- getToaYaps(synced_dat,hydros_yaps,rbi_min,rbi_max)

inp <- getInp(hydros_yaps, toa, E_dist="Mixture", n_ss=2,pingType='rbi',
              sdInits=1,rbi_min=rbi_min,rbi_max=rbi_max,ss_data_what="est",
              ss_data=0)
yaps_out <- runYaps(inp,silent=TRUE)
# gives warnings about NaNs produced?
# and the track is quite short.
# what if I give it only triplerx? No warnings, and the track is longer.
# And when I give it the longer dataset, even though it loses some crap
# to 11, the overall track is actually longer and looks pretty good.
plotYaps(inp=inp,yaps_out=yaps_out, type="map")

# Write that out to a csv
pl <- yaps_out$pl
pl$X <- pl$X + inp$inp_params$Hx0
pl$Y <- pl$Y + inp$inp_params$Hy0
pl$top <- pl$top + inp$inp_params$T0

d <- data.frame( x = yaps_out$pl$X + inp$inp_params$Hx0,
                 y = yaps_out$pl$Y + inp$inp_params$Hy0,
                 tnum = yaps_out$pl$top + inp$inp_params$T0 )
write.csv(d,"track-7adb.csv")
  