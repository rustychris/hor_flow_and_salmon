// STAN model for pseudo-range multilateraion
// of a single ping.
data {
 int<lower=1> Nb; // number of beacons
 
 real rx_t[Nb];
 real rx_x[Nb];
 real rx_y[Nb];
 
 real<lower=0.0> sigma_t;
 real<lower=0.0> sigma_x;
 real<lower=0.0> rx_c[Nb];
}
parameters {
 // only sample the unconstrained time shifts
 real x;
 real y;
}
transformed parameters {
  real dt[Nb];
  for(a in 1:Nb) {
    dt[a]=sqrt( square(rx_x[a] - x) + square(rx_y[a]-y)) / rx_c[a];
  }
}
model {
  real tdoa;

  x ~ normal(0,sigma_x);
  y ~ normal(0,sigma_x);
  
  for ( a in 1:Nb ) {
    for ( b in (a+1):Nb ) {
      // how much later b heard it than a
      tdoa=rx_t[b]-rx_t[a];
      
      tdoa ~ normal(dt[b]-dt[a],sigma_t);
    }
  }
}


