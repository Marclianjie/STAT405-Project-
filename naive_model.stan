data {
  int<lower=1> N;                        // number of observations
  array[N] int<lower=1, upper=7> y;      // observed happiness ratings
}

transformed data {
  vector[6] c;                           // fixed cutpoints for 7 categories
  c[1] = -2.5;
  c[2] = -1.5;
  c[3] = -0.5;
  c[4] =  0.5;
  c[5] =  1.5;
  c[6] =  2.5;
}

parameters {
  real mu;                               // global latent mean
  real<lower=0> sigma;                   // latent SD
  vector[N] z;                           // non-centered latent variables
}

transformed parameters {
  vector[N] x;                           // latent continuous happiness
  x = mu + sigma * z;
}

model {
  // Priors
  mu ~ normal(0, 1.5);
  sigma ~ exponential(1);
  z ~ normal(0, 1);

  // Likelihood
  y ~ ordered_logistic(x, c);
}

generated quantities {
  array[N] real log_lik;
  array[N] int<lower=1, upper=7> y_rep;

  real x_new;                            // new latent happiness draw
  int<lower=1, upper=7> y_new;           // predicted new observed happiness rating

  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | x[n], c);
    y_rep[n] = ordered_logistic_rng(x[n], c);
  }

  // Posterior predictive for one new observation
  x_new = normal_rng(mu, sigma);
  y_new = ordered_logistic_rng(x_new, c);
  
}
