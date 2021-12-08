// drive model (normal dist)
data {
  int<lower=1> Nseg;            // number of segments
  real seg_swim_urel[Nseg];        // long swim velocity
  real seg_swim_lat[Nseg];         // lat swim speed
  real seg_hydro_speed[Nseg];         // lat swim speed
}
parameters {
  real m_hydro; // slope of linear relationship swim_urel ~ hydro
  real m_lat; // slope of linear relationship swim_urel ~ swim_lat
  real inter;  // intercept of linear model
}

model {
  // weak priors
  m_hydro ~ normal(0,1); 
  m_lat ~ normal(0,1); 
  inter ~ normal(0,1); 

  // likelihood
  for(s in 1:Nseg) {
    seg_swim_urel[s] ~ normal( m_hydro*seg_hydro_speed[s]
                               + m_lat*seg_swim_lat[s]
                               + inter,
                               0.1);
  }
}
