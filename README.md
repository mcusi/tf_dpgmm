# tf_dpgmm
Variational inference in Dirichlet process Gaussian mixture model (tensorflow implementation), for spherical and diagonal covariance models 

There is a folder for each model type, and each contains:
- a pdf containing the equations and derivations for the evidence lower bound and variational updates
- ```dpgmm_vi.py```: a tensorflow implementation of variational inference in the model 
- ```bound_check.py```: a comparison of the analytical and Monte Carlo estimates of the ELBO. We used this to check that our derivations and code are correct, because the two estimates match. 
- ```demos.py```: examples of how to use ```dpgmm_vi.py```, including plotting changes in ELBO with each update and clustering results

These codes have not been optimized for performance. Please let us know if you find any mistakes!
