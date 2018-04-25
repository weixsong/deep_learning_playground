# Reference

https://gist.github.com/wuaalb/c5b85d0c257d44b0d98a
https://github.com/casperkaae/parmesan/issues/22


"""
Reproducing the experiment of section 6.1 "Representative Power of Normalizing Flows" from [1]_ (figure 3).
This experiment visually demonstrates that normalizing flows can successfully transform a simple 
initial approximate distribution q_0(z) to much better approximate some known non-Gaussian bivariate 
distribution p(z).
The known target distributions are specified using energy functions U(z) (table 1 from [1]_).
p(z) = \frac{1}{Z} e^{-U(z)}, where Z is the unknown partition function (normalization constant); 
that is, p(z) \prop e^{-U(z)}.
Steps
-----
1. Generate samples from initial distribution z0 ~ q_0(z) = N(z; \mu, \sigma^2 I).
   Here \mu and \sigma can either be fixed to something "reasonable", or estimated as follows.
   Draw auxillary random variable \eps from standard normal distribution
     \eps ~ N(0, I)
   and apply linear normalizing flow transformation f(\eps) = \mu + \sigma \eps, reparameterizing 
   \sigma = e^{1/2*log_var} to ensure \sigma > 0, then jointly optimize {mu, log_var} together 
   with the other normalizing flow parameters (see below).
2. Transform the initial samples z_0 through K normalizing flow transforms, from which we obtain the 
   transformed approximate distribution q_K(z),
     log q_K(z) = log q_0(z) - sum_{k=1}^K log det |J_k|
   where J_k is the Jacobian of the k-th (invertible) normalizing flow transform.
   E.g. for planar flows,
     log q_K(z) = log q_0(z) - sum_{k=1}^K log |1 + u_k^T \psi_k(z_{k-1})|
   where each flow includes model parameters \lambda = {w, u, b}.
3. Jointly optimize all model parameters by minimizing KL-divergence between the approxmate distribution q_K(z)
   and the true distribution p(z).
     loss = KL[q_K(z)||p(z)] = E_{z_K~q_K(z)} [log q_K(z_K) - log p(z_K)]
                             = E_{z_K~q_K(z)} [(log q_0(z_0) - sum_k log det |J_k|) - (-U(z_K) - log(Z))]
                             = E_{z_0~q_0(z)} [log q_0(z_0) - sum_k log det |J_k| + U(f_1(f_2(..f_K(z_0)))) + log(Z)]
   Here the partition function Z is independent of z_0 and model parameters, so we can ignore it for the optimization
     \frac{\partial}{\partial params} loss \prop \frac{\partial}{\partial params} E_{z_0~q_0(z)} [log q0(z0) - sum_k log det |J_k| + U(z_K)]
References
----------
   ..   [1] Jimenez Rezende, D., Mohamed, S., "Variational Inference with Normalizing Flows", 
            Proceedings of the 32nd International Conference on Machine Learning, 2015.
"""