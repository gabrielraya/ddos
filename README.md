# Score-based generative models for unsupervised out-of-distribution detection

**Problem**. The model likelihood of probabilistic models for high dimensional distributions suffers from the curse of
dimensionality, assigning sometimes higher likelihood values to out-of-distribution (OOD) samples. A possible
explanation is the existence of a mismatch of the typical set and the regions with the highest likelihood.


*Can we extract the typical set using score based Markov Chain (MC) samplers?*

**Method**

The goal of MC samplers is to sample from the typical set Here, we first
1. Train a Score based Generative Model on in distribution data i e MNIST data)
2. Use a score based MC sampler (i e Langevin dynamics) to get samples from the “typical set”
3.Map the high dimensional space to a 1 d dimensional space by
1.Computing the mean of the norm of the multiscale score
3. Do density estimation on 1 d (avoiding the curse of dimensionality)





## To do 

1. Retrained the model to achieve the mentioned with inception score and fid, and likelihood (bits/dim).





# Testing Libraries

Because we work on experimental research we must test our hypothesis. 
For that we use the amazing libraries that allow us to easily test our ideas:

1. absl 
2. [ML Collections](https://github.com/google/ml_collections) : Library of Python Collections designed for ML use cases.