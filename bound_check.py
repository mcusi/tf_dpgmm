import numpy as np
from scipy.stats import norm, gamma, rv_discrete, beta
from scipy.special import digamma
from scipy.special import gamma as gammaf

"""
Code to check analytical derivations of ELBO in Variational_Inference_in_DPGMM_derivation.pdf against Monte Carlo estimates 

python bound_check.py

mcusi@mit.edu, july 2018
"""

np.random.seed(0)

D=1
K=2

nu = np.random.randn(K)
zeta = np.array([0.2, 0.8])

a = np.ones(K)
b = 1.5*np.ones(K)

lambda1 = np.ones(K)
lambda2 = 2.*np.ones(K)

alpha = 3.
p_phi = beta(1, alpha)
p_mu = norm
p_tau = gamma(a=1., scale=1.)
def log_p_z(phi, z):
	p = np.concatenate([[1], np.cumprod(1-phi[:-1])]) * phi
	return np.log(p[z])
def p_x(z, mu, tau): return norm(loc=mu[z], scale=np.sqrt(1./tau[z]))

q_phi = beta(lambda1, lambda2)
q_mu = norm(loc=nu)
q_tau = gamma(a=a, scale=1./b) #tau is precision!
q_z = rv_discrete(values=(range(K), zeta))

x = 3.

N = 20001
# ####### phi term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
lgamma = lambda x: np.log(gammaf(x))
bound = sum((lgamma(1. + alpha) - lgamma(alpha)
			+ (alpha - 1.)*(digamma(l2_k) - digamma(l1_k + l2_k))
			- lgamma(l1_k + l2_k) + lgamma(l1_k) + lgamma(l2_k)
			- (l1_k - 1.)*(digamma(l1_k) - digamma(l1_k + l2_k))
			- (l2_k - 1.)*(digamma(l2_k) - digamma(l1_k + l2_k)))
			for (l1_k, l2_k) in zip(lambda1, lambda2))
print("Analytical phi term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of phi term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	phi = q_phi.rvs()

	bounds.append(sum(p_phi.logpdf(phi) - q_phi.logpdf(phi))) #Sum over K for MC estimate
	if i%5000 == 0: print(i, np.mean(bounds))

# ####### mu term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
bound = sum(-0.5*nu_k**2 for nu_k in nu)
print("Analytical mu term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of mu term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	mu = q_mu.rvs()

	bounds.append(sum(p_mu.logpdf(mu) - q_mu.logpdf(mu))) #Sum over K for MC estimate
	if i%5000 == 0: print(i, np.mean(bounds))

# ####### tau term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
bound = sum(lgamma(a_k) - (a_k-1.)*digamma(a_k) - np.log(b_k) + 1 - np.divide(a_k,b_k)
			for (a_k, b_k) in zip(a, b))
print("Analytical tau term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of mu term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	tau = q_tau.rvs()

	bounds.append(sum(p_tau.logpdf(tau) - q_tau.logpdf(tau))) #Sum over K for MC estimate
	if i%5000 == 0: print(i, np.mean(bounds))

# ####### z term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
bound = sum(zeta_k*(
				- np.log(zeta_k)
				+ digamma(l1_k) - digamma(l1_k+l2_k)
				+ sum(digamma(lambda2[j]) - digamma(lambda1[j]+lambda2[j]) for j in range(k)))
			for (l1_k, l2_k, zeta_k, k) in zip(lambda1, lambda2, zeta, range(K)))
print("Analytical z term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of z term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	phi = q_phi.rvs()
	z = q_z.rvs()

	bounds.append(log_p_z(phi, z) - q_z.logpmf(z)) #There's only a single datapoint, so no need for sum
	if i%5000 == 0: print(i, np.mean(bounds))

# ####### x term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
bound = sum(zeta_k * (
			-D/2. * (np.log(2 * np.pi) - digamma(ak) + np.log(bk))
			-(ak/(2.*bk)) * (x - nu_k)**2
			-(ak/(2.*bk)) * gammaf(3./2.) * (2*np.pi)**(-D/2.) * 2.**(3./2.)
		) for (ak, bk, nu_k, zeta_k) in zip(a, b, nu, zeta))
print("Analytical x term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of x term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	mu = q_mu.rvs()
	tau = q_tau.rvs()
	z = q_z.rvs()

	bounds.append(p_x(z, mu, tau).logpdf(x)) #There's only a single datapoint, so no need for sum
	if i%5000 == 0: print(i, np.mean(bounds))