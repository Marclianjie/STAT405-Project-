data {
  int<lower=1> N;       
  array[N] int<lower=1, upper=7> y;  
}

transformed data {
  vector[6] c;                     
  c[1] = -2.5;
  c[2] = -1.5;
  c[3] = -0.5;
  c[4] =  0.5;
  c[5] =  1.5;
  c[6] =  2.5;
}

parameters {
  real mu;                             
  real<lower=0> sigma;              
  vector[N] z;                       
}

transformed parameters {
  vector[N] x;                     
  x = mu + sigma * z;
}

model {
  mu ~ normal(0, 1.5);
  sigma ~ exponential(1);
  z ~ normal(0, 1);
  y ~ ordered_logistic(x, c);
}

generated quantities {
  array[N] real log_lik;
  array[N] int<lower=1, upper=7> y_rep;

  real x_new;                          
  int<lower=1, upper=7> y_new;          

  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | x[n], c);
    y_rep[n] = ordered_logistic_rng(x[n], c);
  }

  x_new = normal_rng(mu, sigma);
  y_new = ordered_logistic_rng(x_new, c);
}

