//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=7> y;
}

parameters {
  vector[N] h_raw;
  real<lower=0> tau;
  ordered[6] c;
}

transformed parameters {
  vector[N] h;
  h = tau * h_raw;
}

model {
  h_raw ~ normal(0, 1);
  tau ~ exponential(1);
  c ~ normal(0, 1.5);

  y ~ ordered_logistic(h, c);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | h[n], c);
    y_rep[n] = ordered_logistic_rng(h[n], c);
  }
}


