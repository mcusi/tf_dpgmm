# tf_dpgmm (spherical covariance)
A note on the form of Q: the distribution over mu has spherical variance equal to 1, which may not be desirable because it means that the uncertainty over mu in the posterior is always the same rather than updating in response to the data. This is fixed in the diagonal derivation.

If any datapoints are equal to the zero vector, they will be ignored. See the use of ```zeta_mask``` in ```dpgmm_vi.py```. This enables the use of differently sized datasets (because you can pad the smaller ones with zero vectors), but may not be what you want!