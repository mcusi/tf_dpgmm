# tf_dpgmm
Variational inference in Dirichlet process isotropic Gaussian mixture model (tensorflow implementation)

Our equations and derivations for the evidence lower bound and variational updates are included in the pdf. We implemented these in ```dpgmm_vi.py```.
To show these are correct, we compared the analytical bounds to Monte Carlo estimates of the ELBO (run ```python bound_check.py```).
You can see examples of how to use ```dpgmm_vi.py``` by running ```python demos.py```, which will output plots of the change in the ELBO over time and clustering results.

Please let us know if you find any mistakes!
