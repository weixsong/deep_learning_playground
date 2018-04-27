# Loss of the Normalizing flow for 6.1, the synthetic data

Let $$p(z)$$ be the true distribution of bivariate distribution.
and, $$z0 \sim q_0(z)$$ is a simple distribution than we already know, then we need to transform this simple distribution by Normalizing Flow to approximate the true distribution $$p(z)$$ that we want to get.

We use KL divergence to measure the distant of our learned distribution with the true distribution.
$$
KL(q_k(z_k)||p(z)) = \\
E_{z_k \sim q_k(z_k)}[logq_k(z_k) - logp(z_k)] = \\
E_{z_k \sim q_k(z_k)}[logq_0(z_0) - sum_k logdet(Jacobian) - (-U(z_K) - log(Z))] = \\
E_{z_0 \sim q_0(z)} [log q_0(z_0) - sum_k log det |J_k| + U(z_k) + log(Z)]
$$

