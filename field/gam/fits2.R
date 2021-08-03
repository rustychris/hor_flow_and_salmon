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

# raw data as output by track_analyses.py
seg_data <- read.csv('../hor_yaps/segment_correlates.csv')
# convert tag id to a factor for use as random effect in mgcv
seg_data$tag <- factor(seg_data$id)
seg_data$waterdepth <- seg_data$model_z_eta - seg_data$model_z_bed
seg_data$vor <- abs(seg_data$model_vor_top2m)

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
summary(mod_segs)
#visreg(mod_segs) 
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
                gamma=5)
summary(mod_segs_ar)
print(paste("Max EDF: ",max(summary(mod_segs_ar)$edf)))
# Now vorticitiy is a clear loser.
# at gamma=2, turbidity is dropped.
# at gamma=5, waterdepth, and reach_velo_ms also get dropped.
options(repr.plot.width=6,repr.plot.height=3.0)
Yl=c(-0.5,0.5)
Y <- coord_cartesian(ylim=Yl)
pnt<-list(alpha=0.2,size=0.4)
p1<-visreg(mod_segs_ar,"swim_lat",fill=list(fill="green"), points=pnt,gg=TRUE) + coord_cartesian(ylim=Yl,xlim=c(0,0.5))
p2<-visreg(mod_segs_ar,"hydro_speed",fill=list(fill="green"),points=pnt,gg=TRUE) + Y
p3<-visreg(mod_segs_ar,"waterdepth",fill=list(fill="green"),points=pnt,gg=TRUE) + Y
#p4<-visreg(mod_segs_ar,"tag",gg=TRUE)
p5<-visreg(mod_segs_ar,"hour",fill=list(fill="green"), points=pnt,gg=TRUE) + Y
#p6<-visreg(mod_segs_ar,"turb",fill=list(fill="green"), points=list(alpha=0.4),gg=TRUE) + Y
#p7<-visreg(mod_segs_ar,"reach_velo_ms",fill=list(fill="green"), points=list(alpha=0.4),gg=TRUE) + Y

pan<-grid.arrange(p1,p2,p3,p5,nrow=2)
ggsave('mod3_segs_lon.png',plot=pan,width=6,height=4.0)

######################### 

mod_lat <- bam(swim_lat ~ s(hydro_speed,bs="cs") 
                + s(swim_urel,bs="cs")
                + s(vor,bs="cs")
                + s(waterdepth,bs="cs")
                #+ s(tag,bs='re')
                + s(hour,bs="cc") 
                + s(turb,bs='cs')
                + s(reach_velo_ms,bs='cs'),
                data=mod_data,knots=list(hour=c(0,24)),
                weights=mod_data$weight,
                gamma=1)
summary(mod_lat)


acf(resid(mod_lat), main="acf(resid(mod_lat))")
lat_r1 <- start_value_rho(mod_lat, plot=TRUE)

# Use bam in order for the AR options 
# to take effect.
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
                   gamma=8)
summary(mod_lat_ar)
print(paste("Max EDF: ",max(summary(mod_lat_ar)$edf)))

# gamma=1: vorticity is a clear loser.
# gamma=5: turbidity is dropped.
# gamma=10: drops hydro_speed

options(repr.plot.width=6,repr.plot.height=5.5)
Yl=c(-0.1,0.5)
Y <- coord_cartesian(ylim=Yl)
p1<-visreg(mod_lat_ar,"swim_urel",fill=list(fill="green"), points=pnt,gg=TRUE) + coord_cartesian(ylim=Yl,xlim=c(-0.5,0.5))
# p2<-visreg(mod_lat_ar,"hydro_speed",fill=list(fill="green"),points=list(alpha=0.4),gg=TRUE) + Y
#p3<-visreg(mod_lat_ar,"waterdepth",fill=list(fill="green"),points=pnt,gg=TRUE) + Y
p5<-visreg(mod_lat_ar,"hour",fill=list(fill="green"), points=pnt,gg=TRUE) + Y
#p6<-visreg(mod_lat_ar,"turb",fill=list(fill="green"), points=list(alpha=0.4),gg=TRUE) + Y
p7<-visreg(mod_lat_ar,"reach_velo_ms",fill=list(fill="green"), points=pnt,gg=TRUE) + Y

pan<-grid.arrange(p1,p5,p7,nrow=1)
ggsave('mod3_segs_lat.png',plot=pan,width=6,height=2.5)

##### 

# lateral swimming is non-negative, so could argue that log-transform
# or similar is appropriate.  As a quick test, this does not change the
# results substantially, but makes the results harder to understand.

mod_data$log_lat <- log(mod_data$swim_lat)
mod_llat <- bam(log_lat ~ s(hydro_speed,bs="cs") 
               + s(swim_urel,bs="cs")
               + s(vor,bs="cs")
               + s(waterdepth,bs="cs")
               #+ s(tag,bs='re')
               + s(hour,bs="cc") 
               + s(turb,bs='cs')
               + s(reach_velo_ms,bs='cs'),
               data=mod_data,knots=list(hour=c(0,24)),
               gamma=1)
summary(mod_llat)


acf(resid(mod_llat), main="acf(resid(mod_llat))")
llat_r1 <- start_value_rho(mod_llat, plot=TRUE)

# Seems I have to use bam in order for the AR options 
# to take effect.
mod_llat_ar <- bam(log_lat ~ s(hydro_speed,bs="cs") 
                  + s(swim_urel,bs="cs")
                  + s(vor,bs="cs")
                  + s(waterdepth,bs="cs")
                  #+ s(tag,bs='re')
                  + s(hour,bs="cc") 
                  + s(turb,bs='cs')
                  + s(reach_velo_ms,bs='cs'),
                  rho=lat_r1, AR.start=mod_data$start.event,
                  data=mod_data,
                  gamma=10)
summary(mod_llat_ar)

options(repr.plot.width=6,repr.plot.height=5.5)
Yl=c(-4,1)
Y <- coord_cartesian(ylim=Yl)
p1<-visreg(mod_llat_ar,"swim_urel",fill=list(fill="green"), points=pnt,gg=TRUE) + coord_cartesian(ylim=Yl,xlim=c(-0.5,0.5))
# p2<-visreg(mod_llat_ar,"hydro_speed",fill=list(fill="green"),points=list(alpha=0.4),gg=TRUE) + Y
p3<-visreg(mod_llat_ar,"waterdepth",fill=list(fill="green"),points=pnt,gg=TRUE) + Y
p5<-visreg(mod_llat_ar,"hour",fill=list(fill="green"), points=pnt,gg=TRUE) + Y
#p6<-visreg(mod_lat_ar,"turb",fill=list(fill="green"), points=list(alpha=0.4),gg=TRUE) + Y
p7<-visreg(mod_llat_ar,"reach_velo_ms",fill=list(fill="green"), points=pnt,gg=TRUE) + Y

pan<-grid.arrange(p1,p3,p5,p7,nrow=2)

