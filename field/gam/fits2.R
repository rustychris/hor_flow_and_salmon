# follow m-clark.github.io intro, use mgcv
# fits2 tries to combine the inter- and intra- track
# fits into a single operation.
library(mgcv)
library(ggplot2)
library(GGally)
library(visreg)
library(dplyr)
library(gridExtra)
library(itsadug)

# Choice of vertical averaging:
# The gamma values come out different for different vavg.
# The lateral analysis is nearly identical, just with a shift
# in the swim_urel distribution as expected.
# longitudinal analysis is twitchy. With davg velocity, turbidity
# makes it in (!), and with top1m, reach_velo_ms makes it in. 
vavg<-'davg'

# raw data as output by track_analyses.py
seg_data <- read.csv(paste('../hor_yaps/segment_correlates_',vavg,'.csv',sep=''))
# convert tag id to a factor for use as random effect in mgcv
seg_data$tag <- factor(seg_data$id)
seg_data$waterdepth <- seg_data$model_z_eta - seg_data$model_z_bed
seg_data$vor <- abs(get(paste("model_vor_",vavg,sep=""),seg_data))

cols <- c("tag","swim_urel","swim_lat","hydro_speed","turb","hour","flow_100m3s",
          "vor","reach_velo_ms","waterdepth","tnum")

mod_data <- start_event(seg_data[cols],"tnum",event="tag")

# GAM for instantaneous rheotaxis
# Most of the correlates do not have meaningful variation within a 
# track, for example mean river flow does not change over the course
# of a 20 minute track. A random effect term would thus be confounded
# with the slow-moving parameters. For this reason, we omit the 
# random effect.

# For longitudinal swimming there is still the issue of bias related
# to how quickly individuals exit the array.
# Option A: weight samples as before, 1/nsamples for individual
#  As before, this gives a lot of weight to very short tracks.
# Option B: calculate a mean longitudinal swimming velocity, compare
#  to the river velocity, and weight the samples by the inverse of
#  the expected time in the array. 

tags <- mod_data %>% count(tag)
tags$weight <- 1./tags$n

weights=merge( mod_data, tags, by="tag")$weight
mod_data$weight <- weights/mean(weights)


# Initial model with all potential terms included
mod_segs <- bam(swim_urel ~ s(hydro_speed,bs="cs") 
                 + s(swim_lat,bs="cs")
                 + s(vor,bs="cs")
                 + s(waterdepth,bs="cs")
                 #+ s(tag,bs='re')
                 + s(hour,bs="cc") 
                 + s(turb,bs='cs')
                 + s(reach_velo_ms,bs='cs'),
                 data=mod_data,knots=list(hour=c(0,24)),
                 weights=mod_data$weight,
                 gamma=1)
# summary(mod_segs)
# visreg(mod_segs) 
# inclusion of weights increased p-value for vor and waterdepth,
# but they are still "significant" at this stage.

# Omitting the random tag effect, increasing gamma to 20,
# and using the 'cs' smooth where possible, the edf for 
# vor, waterdepth, and turbidity and reach_velo_ms are reduced 
# to 0. I think it's wrong to force gamma high at this point if
# I'm going to turn around and remove autocorrelation.
# So first fit with gamma=1, get ACF, fit again, and *then* tune
# gamma.

# Following tips here:
# https://cran.r-project.org/web/packages/itsadug/vignettes/acf.html
acf(resid(mod_segs), main="acf(resid(mod_segs))")
r1 <- start_value_rho(mod_segs, plot=TRUE)

# Have to use bam in order for the AR options 
# to take effect.
for ( gamma in 1:20 ) {
  mod_segs_ar <- bam(swim_urel ~ s(hydro_speed,bs="cs") 
                  + s(swim_lat,bs="cs")
                  + s(vor,bs="cs")
                  + s(waterdepth,bs="cs")
                  #+ s(tag,bs='re')
                  + s(hour,bs="cc") 
                  + s(turb,bs='cs')
                  + s(reach_velo_ms,bs='cs'),
                  rho=r1, AR.start=mod_data$start.event,
                  data=mod_data,
                  weights=mod_data$weight,
                  gamma=gamma)
  max_edf<-max(summary(mod_segs_ar)$edf)
  print(paste("Gamma: ",gamma," max EDF: ",max_edf))
  if ( max_edf <= 4.0 ) {
    break
  }
}
mod_segs_ar_sum<-summary(mod_segs_ar)
mod_segs_ar_sum

# Now vorticity is a clear loser.
# at gamma=2, turbidity is dropped.
# at gamma=5, waterdepth, and reach_velo_ms also get dropped.
options(repr.plot.width=6,repr.plot.height=3.0)
Yl=c(-0.5,0.5)
Y <- coord_cartesian(ylim=Yl)
pnt<-list(alpha=0.2,size=0.4)

# then test for any affect across
# choice of velocity.
var_names<-names(mod_segs_ar$var.summary)
panels<-list()
for ( vi in 1:length(var_names) ) {
  vname<-var_names[vi]
  p_val<-mod_segs_ar_sum$s.pv[vi]
  if ( p_val > 0.05 ) { next }
  if ( mod_segs_ar_sum$edf[vi] < 0.5) { next }
  if ( vname=='swim_lat' ) {
      coords <-coord_cartesian(ylim=Yl,xlim=c(0,0.5))
    } else {
      coords<-Y 
    }
  panel<-visreg(mod_segs_ar,vname,fill=list(fill="green"), points=pnt,gg=TRUE) + coords 
  panels<-append(panels,list(panel))
}

pan<-grid.arrange(grobs=panels,nrow=2)
ggsave(paste('mod3_segs_lon',vavg,'.png',sep=''),plot=pan,width=6,height=4.0)

######################### 

mod_lat <- bam(swim_lat ~ s(hydro_speed,bs="cs") 
                + s(swim_urel,bs="cs")
                + s(vor,bs="cs")
                + s(waterdepth,bs="cs")
                + s(hour,bs="cc") 
                + s(turb,bs='cs')
                + s(reach_velo_ms,bs='cs'),
                data=mod_data,knots=list(hour=c(0,24)),
                weights=mod_data$weight,
                gamma=1)

acf(resid(mod_lat), main="acf(resid(mod_lat))")
lat_r1 <- start_value_rho(mod_lat, plot=TRUE)

# Use bam in order for the AR options 
# to take effect.
for ( gamma in 1:20 ) {
  mod_lat_ar <- bam(swim_lat ~ s(hydro_speed,bs="cs") 
                   + s(swim_urel,bs="cs")
                   + s(vor,bs="cs")
                   + s(waterdepth,bs="cs")
                   #+ s(tag,bs='re')
                   + s(hour,bs="cc") 
                   + s(turb,bs='cs')
                   + s(reach_velo_ms,bs='cs'),
                   rho=lat_r1, AR.start=mod_data$start.event,
                   data=mod_data,
                   weights=mod_data$weight,
                   gamma=gamma)
  max_edf<-max(summary(mod_lat_ar)$edf)
  print(paste("Gamma: ",gamma," max EDF: ",max_edf))
  if ( max_edf <= 4.0 ) {
    break
  }
}
mod_lat_ar_sum<-summary(mod_lat_ar)
mod_lat_ar_sum


options(repr.plot.width=6,repr.plot.height=5.5)
Yl=c(-0.1,0.5)
Y <- coord_cartesian(ylim=Yl)

var_names<-names(mod_lat_ar$var.summary)
panels<-list()
for ( vi in 1:length(var_names) ) {
  vname<-var_names[vi]
  p_val<-mod_lat_ar_sum$s.pv[vi]
  if ( p_val > 0.05 ) { next }
  if ( mod_lat_ar_sum$edf[vi] < 0.5) { next }
  if ( vname=='swim_lat' ) {
    coords <-coord_cartesian(ylim=Yl,xlim=c(0,0.5))
  } else if ( vname=='swim_urel' ) {
    coords <- coord_cartesian(ylim=Yl,xlim=c(-0.5,0.5))
  } else {
    coords<-Y 
  }
  panel<-visreg(mod_lat_ar,vname,fill=list(fill="green"), points=pnt,gg=TRUE) + coords 
  panels<-append(panels,list(panel))
}
pan<-grid.arrange(grobs=panels,nrow=1)
ggsave(paste('mod3_segs_lat',vavg,'.png',sep=''),plot=pan,width=6,height=2.5)

# Briefly revisit the possibility of encoding that some predictors are only functions
# of tag:
#  tensor product? Instead of a cs(reach_velo_ms), would the term be
#    te(reach_velo_ms,tag)?
#  No.  te should have two continuous variables, not a factor.
# swim_urel(tag,idx) ~ reach_velo(tag)
# Not seeing a clear path here.

# A bit more reading, and I think I need a better distribution. The qq-plot
# (gam.check(mod_segs_ar)) suggests that the true distribution has heavier tails
# than a normal.

