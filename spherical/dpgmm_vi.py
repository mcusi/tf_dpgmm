import tensorflow as tf
from scipy.special import gamma as gamma_func
import numpy as np
import collections

"""
Spherical COVARIANCE

Unsupervised clustering in R^D

TF implementation of variational inference in a 
Dirichlet Process isotropic Gaussian mixture model
Derivation in Variational_Inference_in_DPGMM_derivation.pdf included in git repo 

Clusters matrix X (batch_size x N x D) of N datapoints with dimensionality D
If a datapoint = zero vector, it is ignored. 
This option allows you to use batched datasets of different sizes

For examples of how to use this code, see demos.py

mcusi@mit.edu, july 2018

"""

class dpgmm():

    ######### INITIALIZATION ##########################################################################################################

    def __init__(self, alpha, D, n_iter, T, covariance_type='diagonal'):

        self.alpha = alpha; #Dirichlet concentration parameter
        self.D = D; self.Dfl = tf.cast(self.D, dtype=tf.float32); #dimensionality of data
        self.T = T; #truncation value
 
        self.gaussian_const = np.divide(gamma_func(1.5)*(2.0**1.5)*self.D, np.sqrt(2.*np.pi))

        #Initialization settings
        self.mu_std = 5.

        #inference settings
        self.n_iter = n_iter;
        self.log_constant = 1e-30

    def initialize_latents(self, X, batch_size, shared=True, use_mask=True):
        """
        > randomly initializes variational distribution parameters
        > if shared == True, batches share the same initialization
        """

        N = tf.shape(X)[1]
        shape_T = [self.T] if shared else [batch_size, self.T]
        shape_TD = [self.T, self.D] if shared else [batch_size, self.T, self.D]

        a = tf.get_variable("a", shape_T, dtype=tf.float32,
                initializer=tf.ones_initializer())
        b = tf.get_variable("b", shape_T, dtype=tf.float32,
                initializer=tf.ones_initializer())
        lambda_1 = tf.get_variable("lambda_1", shape_T, dtype=tf.float32,
                initializer=tf.ones_initializer())
        lambda_2 = tf.get_variable("lambda_2", shape_T, dtype=tf.float32,
                initializer=tf.ones_initializer())
        nu = tf.get_variable("nu", shape_TD, dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=self.mu_std)) 
        omega = tf.get_variable("omega", shape_T, dtype=tf.float32,
                initializer=tf.ones_initializer())

        if shared:

            a = tf.tile(a[tf.newaxis, :], [batch_size, 1])
            b = tf.tile(b[tf.newaxis, :], [batch_size, 1])
            lambda_1 = tf.tile(lambda_1[tf.newaxis, :], [batch_size, 1])
            lambda_2 = tf.tile(lambda_2[tf.newaxis, :], [batch_size, 1])
            nu = tf.tile(nu[tf.newaxis, :, :], [batch_size, 1, 1])
            omega = tf.tile(omega[tf.newaxis, :], [batch_size, 1])
        
        # zeta will be the first in the update distribution 
        # so this initialization is only necessary for ELBO calculation
        alpha_vec = tf.fill([batch_size, self.T], self.alpha)
        zeta_dist = tf.distributions.Dirichlet(alpha_vec)
        #zeta: batch_size N T
        zeta = tf.transpose(zeta_dist.sample([N]),perm=[1,0,2])
        #mask: batch_size N
        if use_mask:
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.equal(X,0),axis=2)), dtype=tf.float32)
        else:
            mask = tf.ones([batch_size, N])

        return a, b, lambda_1, lambda_2, nu, omega, zeta, mask

    ######### UPDATE EQUATIONS ##########################################################################################################

    def update_lambda(self, zeta_mask):
        ##lambda_1: only sum over datapoints
        #nu_z batch N T
        #embedding_weights batch N
        lambda_1 = 1.0 + tf.reduce_sum(zeta_mask, axis=1) #over N
        ##lambda_2: requires sum over classes as well as datapoints 
        #nu_z: batch N T 
        l = tf.cumsum(zeta_mask, axis=2, reverse=True, exclusive=True) #over T
        lambda_2 = self.alpha + tf.reduce_sum(l, axis=1) #over N
        return lambda_1, lambda_2

    def update_nu(self, a, b, zeta_mask, X):
        # nu_z batch N T
        # a batch newaxis T
        # b batch newaxis T
        w = tf.divide(tf.multiply(zeta_mask, 
            a[:, tf.newaxis, :]), b[:, tf.newaxis,:])[:, :, :, tf.newaxis] 
        #w : batch N T newaxis 
        #X : batch N newaxis D
        numer = tf.reduce_sum(tf.multiply(w, X[:, :, tf.newaxis, :]), axis=1) #over N
        denom = 1.0 + tf.reduce_sum(w, axis=1) #over N
        # numer batch T D
        # denom batch T D
        nu = tf.divide(numer,  denom)
        return nu

    def update_omega(self, a, b, zeta_mask):

        # a batch newaxis T
        # b batch newaxis T
        ratio = tf.multiply(tf.divide(a, b), self.gaussian_const)[:, tf.newaxis, :]
        # nu_z batch N T 
        #WANT: omega batch T
        omega = 1.0 + tf.reduce_sum( tf.multiply(zeta_mask, ratio) , axis=1) #over N
        return omega

    def update_ab(self, nu, omega, zeta_mask, X):
        #nu_z_masked batch N T
        #a batch T
        a = 1.0 + tf.multiply(self.Dfl/2.0, tf.reduce_sum(zeta_mask, axis=1))#over N
        
        #X batch N newaxis D
        #nu batch newaxis T D
        #difference norm batch N T        
        difference_norm = tf.reduce_sum(tf.square(X[:,:,tf.newaxis,:] - nu[:,tf.newaxis,:,:]),axis=3)
        #omega batch newaxis T
        s = difference_norm + tf.divide(self.gaussian_const, omega[:, tf.newaxis, :])
        #s batch N T 
        #zeta_mask batch N T
        #b batch T 
        b = 1.0 + 0.5*tf.reduce_sum(tf.multiply(zeta_mask, s), axis=1) #over N
        
        return a, b

    def eta_x(self, a, b, nu, omega, X):
        """
        eta_x_{i, k} = E_q[log P(x_{i} | z_i = k, mu_{k}, var_{k})]

        """

        #a batch_size, T
        #b batch_size, T
        ab1 = tf.multiply(-self.Dfl/2.0, tf.log(2*np.pi) - tf.digamma(a) + tf.log(b))[:,tf.newaxis, :]
        ab2 = tf.divide(a, -2.0*b)[:,tf.newaxis, :]

        # X: batch N newaxis D
        # mu: batch newaxis T D
        norm_difference = tf.reduce_sum(tf.square(tf.subtract(X[:, :, tf.newaxis, :], nu[:, tf.newaxis, :, :])),axis=3)
        #omega: batch newaxis T 
        s = norm_difference + tf.divide(self.gaussian_const, omega[:, tf.newaxis, :])
        #s: batch N T
        #ab1 batch_size, newaxis, T,
        #ab2 batch_size, newaxis, T,
        Eq = ab1 + tf.multiply(ab2, s)

        return Eq

    def eta_z(self, lambda_1, lambda_2):
        #lambda_1, lambda_2: batch_size, T
        d1 = tf.digamma(lambda_1) - tf.digamma(lambda_1 + lambda_2)
        d2 = tf.digamma(lambda_2) - tf.digamma(lambda_1 + lambda_2)
        d_cumsum = tf.cumsum(d2, axis=1, exclusive=True)
        return d1 + d_cumsum

    def update_zeta(self, a, b, lambda_1, lambda_2, nu, omega, X): 

        #self.eta_x: batch N T 
        #self.eta_z: batch newaxis T

        prop_log_zeta = self.eta_z(lambda_1, lambda_2)[:, tf.newaxis, :] - 1. + self.eta_x(a, b, nu, omega, X) 
        #prop_log_nu_z batch N T 
        log_zeta = tf.subtract(prop_log_zeta, tf.reduce_logsumexp(prop_log_zeta, axis=2, keepdims=True)) #over T
        zeta = tf.exp(log_zeta)

        return zeta

    def update_all(self, L, dataset):


        zeta = self.update_zeta(L.a, L.b, L.lambda_1, L.lambda_2, L.nu, L.omega, dataset.X)
        zeta_mask = tf.multiply(dataset.mask[:,:,tf.newaxis],zeta)
        lambda_1, lambda_2 = self.update_lambda(zeta_mask)
        nu = self.update_nu(L.a, L.b, zeta_mask, dataset.X)
        omega = self.update_omega(L.a, L.b, zeta_mask)
        a, b = self.update_ab(nu, omega, zeta_mask, dataset.X) #might have to mess with the order of these

        return a, b, lambda_1, lambda_2, nu, omega, zeta

    ######### VARIATIONAL LOWER BOUND ##########################################################################################################

    def phi_lower_bound_term(self, lambda_1, lambda_2):
        """
        lambda_1: [batch_size, T]
        lambda_2: [batch_size, T]
        """
        term1 =  tf.lgamma(1. + self.alpha) - tf.lgamma(self.alpha)
        term2 = (self.alpha - 1.)*(tf.digamma(lambda_2) - tf.digamma(lambda_1 + lambda_2))
        term3 = -1*tf.lgamma(lambda_1 + lambda_2) + tf.lgamma(lambda_1) + tf.lgamma(lambda_2)
        term4 = tf.multiply(lambda_1 - 1., tf.digamma(lambda_1) - tf.digamma(lambda_1 + lambda_2))
        term5 = tf.multiply(lambda_2 - 1., tf.digamma(lambda_2) - tf.digamma(lambda_1 + lambda_2))
        #sum over clusters
        vb = tf.reduce_sum(term1 + term2 + term3 - term4 - term5, axis=1)
        return vb

    def mu_lower_bound_term(self, nu, omega):
        #nu: [batch_size, T, D]
        #omega [batch_size, T]
        tot = tf.multiply(-self.Dfl/2.0, tf.divide(1.0, omega) + tf.log(omega) - 1.0) - 0.5*tf.reduce_sum(tf.square(nu),axis=2)
        vb = tf.reduce_sum(tot, axis=1) #over number of clusters
        return vb

    def tau_lower_bound_term(self, a, b):
        #a, b: [batch_size, T]
        tot = tf.lgamma(a) - tf.multiply(a - 1.,tf.digamma(a)) - tf.log(b) + a - tf.divide(a, b)
        vb = tf.reduce_sum(tot, axis=1) #sum over clusters 
        return vb

    def z_lower_bound_term(self, lambda_1, lambda_2, zeta, mask):
        #lambda_1: [batch_size, T]
        #lambda_2: [batch_size, T]
        #zeta: [batch_size, N, T]
        c = -tf.log(zeta + self.log_constant) + self.eta_z(lambda_1, lambda_2)[:,tf.newaxis,:]

        # batch_Size N T --> batch N 
        e = tf.reduce_sum(tf.multiply(zeta, c),axis=2) #over clusters
        e_mask = tf.multiply(e, mask)
        vb = tf.reduce_sum(e_mask, axis=1) #over I 

        return vb

    def x_lower_bound_term(self, a, b, nu, omega, zeta_mask, X):
        #X: batch_size, N, D
        #self.eta_x: batch N T 
        EqLogPxGivenZ = self.eta_x(a, b, nu, omega, X)
        #zeta_mask: batch_size, N, T
        tot = tf.multiply(zeta_mask, EqLogPxGivenZ)
        #tot batch_size N T
        vb = tf.reduce_sum(tot, axis=[1,2])
        return vb

    def evidence_lower_bound(self, L, D):
        phi_lb = self.phi_lower_bound_term(L.lambda_1, L.lambda_2) 
        mu_lb = self.mu_lower_bound_term(L.nu, L.omega) 
        tau_lb = self.tau_lower_bound_term(L.a, L.b)
        z_lb = self.z_lower_bound_term(L.lambda_1, L.lambda_2, L.zeta, D.mask)
        x_lb = self.x_lower_bound_term(L.a, L.b, L.nu, L.omega, tf.multiply(D.mask[:,:,tf.newaxis],L.zeta), D.X)
        return phi_lb + mu_lb + tau_lb + z_lb + x_lb

    ######### INFERENCE FUNCTIONS ##########################################################################################################

    def infer(self, _a, _b, _lambda_1, _lambda_2, _nu, _omega, _zeta, X, mask):
        """
        Performs variational inference in DPGMM for n_iter number of iterations,
        then returns inferred latent variables

        _a, _b, _lambda_1, _lambda_2, _nu, _zeta: initial parameters for inference
        X: data matrix (batch_size x nDatapoints x dimensions)
        mask: 1 if consider as datapoint, 0 if ignore (batch_size x nDatapoints)
        """
            
        ##Initial input into "while" loop, i.e., inference iterations
        i = tf.constant(0)
        latents = collections.namedtuple('latents', ['a', 'b', 'lambda_1', 'lambda_2', 'nu', 'omega', 'zeta'])
        dataset = collections.namedtuple('dataset', ['X', 'mask'])
        init_iteration = (i, latents(_a, _b, _lambda_1, _lambda_2, _nu, _omega, _zeta), dataset(X, mask))

        cond = lambda i, L, D: i < self.n_iter
        def body(i, L, D):
            a, b, lambda_1, lambda_2, nu, omega, zeta = self.update_all(L, D)              
            return (i + 1, latents(a, b, lambda_1, lambda_2, nu, omega, zeta), D)
        
        final_iteration = tf.while_loop(cond, body, init_iteration)

        return final_iteration[1]   

    def elbo_infer(self, _a, _b, _lambda_1, _lambda_2, _nu, _omega, _zeta, X, mask, batch_size):
        """
        Performs variational inference in DPGMM for n_iter number of iterations,
        and also calculates the change in ELBO at each update
        returns inferred latent variables and changes in ELBO 
        """

        i = tf.constant(0)
        latents = collections.namedtuple('latents', ['a', 'b', 'lambda_1', 'lambda_2', 'nu', 'omega', 'zeta'])
        dataset = collections.namedtuple('dataset', ['X', 'mask'])
        #ELBO term names: "updated-variable_term-of-lower-bound"
        ELBO_terms = ['zeta_z', 'zeta_x', 'lambda_phi', 'lambda_z', 'nu_mu', 'nu_x', 'omega_mu', 'omega_x', 'ab_tau', 'ab_x', 'total']
        empty_ELBO_terms = tuple([tf.TensorArray(dtype=tf.float32, size=self.n_iter, element_shape=batch_size, name=ELBO_terms[j]) for j in range(11)])
        init_iteration = (i, latents(_a, _b, _lambda_1, _lambda_2, _nu, _omega, _zeta), dataset(X, mask)) + empty_ELBO_terms

        cond = lambda i, L, D, zeta_z, zeta_x, lambda_phi, lambda_z, nu_mu, nu_x, omega_mu, omega_x, ab_tau, ab_x, total: i < self.n_iter 
        def body(i, L, D, zeta_z, zeta_x, lambda_phi, lambda_z, nu_mu, nu_x, omega_mu, omega_x, ab_tau, ab_x, total):

            zeta = self.update_zeta(L.a, L.b, L.lambda_1, L.lambda_2, L.nu, L.omega, D.X)
            zeta_mask = tf.multiply(D.mask[:,:,tf.newaxis], zeta)
            zeta_z = zeta_z.write(i, self.z_lower_bound_term(L.lambda_1, L.lambda_2, zeta, D.mask)-self.z_lower_bound_term(L.lambda_1, L.lambda_2, L.zeta, D.mask))
            zeta_x = zeta_x.write(i, self.x_lower_bound_term(L.a, L.b, L.nu, L.omega, zeta_mask, D.X)-self.x_lower_bound_term(L.a, L.b, L.nu, L.omega, tf.multiply(D.mask[:,:,tf.newaxis],L.zeta), D.X))

            lambda_1, lambda_2 = self.update_lambda(zeta_mask)
            lambda_phi = lambda_phi.write(i, self.phi_lower_bound_term(lambda_1, lambda_2) - self.phi_lower_bound_term(L.lambda_1, L.lambda_2))
            lambda_z = lambda_z.write(i,self.z_lower_bound_term(lambda_1, lambda_2, zeta, D.mask)-self.z_lower_bound_term(L.lambda_1, L.lambda_2, zeta, D.mask))

            nu = self.update_nu(L.a, L.b, zeta_mask, D.X)
            nu_mu = nu_mu.write(i, self.mu_lower_bound_term(nu, L.omega)-self.mu_lower_bound_term(L.nu, L.omega))
            nu_x = nu_x.write(i, self.x_lower_bound_term(L.a, L.b, nu, L.omega, zeta_mask, D.X)-self.x_lower_bound_term(L.a, L.b, L.nu, L.omega, zeta_mask, D.X))

            omega = self.update_omega(L.a, L.b, zeta_mask)
            omega_mu = omega_mu.write(i, self.mu_lower_bound_term(nu, omega)-self.mu_lower_bound_term(nu, L.omega))
            omega_x = omega_x.write(i, self.x_lower_bound_term(L.a, L.b, nu, omega, zeta_mask, D.X)-self.x_lower_bound_term(L.a, L.b, nu, L.omega, zeta_mask, D.X))            

            a, b = self.update_ab(nu, omega, zeta_mask, D.X)
            ab_tau = ab_tau.write(i, self.tau_lower_bound_term(a, b) - self.tau_lower_bound_term(L.a, L.b))
            ab_x = ab_x.write(i, self.x_lower_bound_term(a, b, nu, omega, zeta_mask, D.X)-self.x_lower_bound_term(L.a, L.b, nu, omega, zeta_mask, D.X))

            updated_L = latents(a, b, lambda_1, lambda_2, nu, omega, zeta)
            total = total.write(i, self.evidence_lower_bound(updated_L, D) - self.evidence_lower_bound(L, D))

            return (i+1, updated_L, D, zeta_z, zeta_x, lambda_phi, lambda_z, nu_mu, nu_x, omega_mu, omega_x, ab_tau, ab_x, total)           

        final_iteration = tf.while_loop(cond, body, init_iteration)

        return final_iteration[1], [final_iteration[i].stack() for i in range(3,14)]

def variational_inference(data, alpha=1.0, T=10, n_iter=10, tf_seed=None, get_elbo=False, tf_device='/cpu:0'):
    """
    Tensorflow setup to run variational inference

    data: matrix of datapoints, size should be (batch_size, max_number_of_datapoints, dimesionality_of_data)
          batches that have different number of datapoints can be run together by padding the smaller data matrices with zero vectors 
    alpha: Dirichlet concentration parameter
    T: truncation paper
    n_iter: number of iterations to run VI for
    get_elbo: if True, measure & return the change in ELBO for each update

    """

    #size of dataset
    batch_size = np.shape(data)[0]
    N = np.shape(data)[1]
    D = np.shape(data)[2]

    with tf.Graph().as_default():
        with tf.device(tf_device):

            tf.set_random_seed(tf_seed)
            X = tf.placeholder(tf.float32, shape=[batch_size, N, D])            

            mixture_model = dpgmm(alpha, D, n_iter, T)
            init_a, init_b, init_lambda_1, init_lambda_2, init_nu, init_omega, init_zeta, mask = mixture_model.initialize_latents(X, batch_size, shared=False)

            if not get_elbo:
                inferred_latents = mixture_model.infer(init_a, init_b, init_lambda_1, init_lambda_2, init_nu, init_omega, init_zeta, X, mask)
            else:
                inferred_latents, ELBO_deltas = mixture_model.elbo_infer(init_a, init_b, init_lambda_1, init_lambda_2, init_nu, init_omega, init_zeta, X, mask, batch_size)

        ##Run graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            
            sess.run(tf.global_variables_initializer())
            if not get_elbo:
                inferred_latents_out = sess.run([inferred_latents], feed_dict = {X: data})
                return inferred_latents_out[0]
            else:
                inferred_latents_out, ELBO_deltas_out = sess.run([inferred_latents, ELBO_deltas], feed_dict = {X: data})
                ELBO_terms = collections.namedtuple('ELBO_terms', ['zeta_z', 'zeta_x', 'lambda_phi', 'lambda_z', 'nu_mu', 'nu_x', 'omega_mu', 'omega_x', 'ab_tau', 'ab_x', 'total'])
                return inferred_latents_out, ELBO_terms(*ELBO_deltas_out)
