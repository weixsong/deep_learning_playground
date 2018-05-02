# Variational Inference with Normalizing Flows
Implementation of paper [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) section 6.1 experiments.

The distributions that we want to learn is as followings:
<img src="images/data.png">
We want to learning the Bi-variate distribution of Z(Z1, Z2).
<img src="images/U_dist.png">
Equation for each of the distribution could be found at paper.



# TODO
* Add IAF

# Reference

[vae-normflow](https://github.com/16lawrencel/vae-normflow)

[Reproduce results from sec. 6.1 in "Variational inference using normalizing flows" ](https://github.com/casperkaae/parmesan/issues/22)

[parmesan/parmesan/layers/flow.py](https://github.com/casperkaae/parmesan/blob/master/parmesan/layers/flow.py)

[wuaalb/nf6_1.py](https://gist.github.com/wuaalb/c5b85d0c257d44b0d98a)