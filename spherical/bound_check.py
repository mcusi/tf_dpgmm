import numpy as np
from scipy.stats import norm, gamma, rv_discrete, beta
from scipy.special import digamma
from scipy.special import gamma as gammaf

lgamma = lambda x: np.log(gammaf(x))

"""
DIAGONAL COVARIANCE

Code to check analytical derivations of ELBO in Variational_Inference_in_DPGMM_derivation.pdf against Monte Carlo estimates 

python bound_check.py

lbh@mit.edu, october 2018
"""

np.random.seed(0)

D=3
K=2

nu = np.random.randn(K, D)
omega = np.random.random([K]) + 1
zeta = np.array([0.2, 0.8])

a = np.ones([K],dtype=np.float32)
b = 1.5*np.ones([K],dtype=np.float32)

lambda1 = np.ones(K,dtype=np.float32)
lambda2 = 2.*np.ones(K,dtype=np.float32)

alpha = 3.
p_phi = beta(1., alpha)
p_mu = norm
p_tau = gamma(a=1., scale=1.)
def log_p_z(phi, z):
	p = np.concatenate([[1], np.cumprod(1-phi[:-1])]) * phi
	return np.log(p[z])
def p_x(z, mu, tau): return norm(loc=mu[z], scale=np.sqrt(1./tau[z]))

q_phi = beta(lambda1, lambda2)
q_mu = norm(loc=nu, scale=1./np.sqrt(omega[:, None]))
q_tau = gamma(a=a, scale=1./b) #tau is precision!
q_z = rv_discrete(values=(range(K), zeta))

x = np.array([1., 2., 3.])

N = 200001

print('Spherical Covariance Model')

# ####### mu term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
# OLD:
# bound = sum(-0.5*nu_k**2 for nu_k in nu)
# NEW:
bound = sum( #over k
	-D/2. * (1./omega + np.log(omega) - 1)
	- 0.5 * (nu**2).sum(axis=1)
)
print("Analytical mu term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of mu term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	mu = q_mu.rvs()
	bounds.append(sum(sum(p_mu.logpdf(mu) - q_mu.logpdf(mu)))) #Sum over K for MC estimate
	if i%5000 == 0: print(i, np.mean(bounds))



# ####### phi term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
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



# ####### tau term in the ELBO ########
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Analytical
#OLD:
# bound = sum(lgamma(a_k) - (a_k-1.)*digamma(a_k) - np.log(b_k) + 1 - np.divide(a_k,b_k)
# 			for (a_k, b_k) in zip(a, b))
#NEW:
bound = sum(lgamma(a_k) - (a_k-1.)*digamma(a_k) - np.log(b_k) + a_k - np.divide(a_k,b_k)
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
bound = sum( #over k
			zeta_k * (
				-D/2. * (np.log(2 * np.pi) - digamma(ak) + np.log(bk))
				-(ak/(2.*bk)) * ((x[None, :] - nu_k)**2).sum(axis=1)
				-(ak/(2.*bk)) * D * gammaf(3./2.) * (2*np.pi)**(-1/2.) * 2.**(3./2.) / omega_k
			)
		for (ak, bk, nu_k, zeta_k, omega_k) in zip(a, b, nu, zeta, omega))
print("Analytical x term in ELBO:", bound)

# Monte Carlo
print('Monte Carlo estimate of x term in ELBO:')
np.random.seed()
bounds = []
for i in range(N):
	mu = q_mu.rvs()
	tau = q_tau.rvs()
	z = q_z.rvs()

	bounds.append(sum(p_x(z, mu, tau).logpdf(x))) #sum over d (There's only a single datapoint, so no need for sum over i)
	if i%5000 == 0: print(i, np.mean(bounds))