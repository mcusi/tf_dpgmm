from dpgmm_vi import variational_inference 
import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import colorsys 

"""
DIAGONAL COVARIANCE

Plots to demonstrate use of dpgmm_vi.py

python demos.py

mcusi@mit.edu, july 2018
"""


def gen_demo_data(batch_size=1, np_seed=None, D=2, use_zeros=True):
    #generates data from multivariate gaussian with diagonal covariance
    Nmax = 6*25
    np.random.seed(np_seed)
    for b in range(batch_size):
        K = np.random.randint(2, high=6+1)
        in_dataset = 0
        for k in range(K):
            mean = 4*np.random.randn(D) - 1
            cov = np.diag(np.random.rand(2) + 0.1)
            n = np.random.randint(10, high=25+1)
            in_dataset += n
            gaussian_data = np.random.multivariate_normal(mean, cov, n)
            _data = gaussian_data if k == 0 else np.vstack((_data, gaussian_data))
        n_zeros = Nmax - in_dataset ##use zeros to pad smaller datasets
        _data = np.vstack((np.zeros((n_zeros,D)),_data))
        np.random.shuffle(_data)
        _data = np.float32(_data[np.newaxis,:,:])
        data = _data if b == 0 else np.concatenate((data,_data))

    return data

def ELBO_demo(np_seed=0, tf_seed=0, alpha=1.0, T=100, max_n_iter=20):

    #Generate toy data
    data = gen_demo_data(batch_size = 1, np_seed = np_seed)
    nonzero_datapoints_batches = np.where(~np.all(data==0, axis=2))

    #Run inference
    inferred_latents, ELBO_deltas = variational_inference(data, alpha=alpha, T=T, n_iter=max_n_iter, tf_seed=tf_seed, get_elbo=True)

    #Plot change in ELBO with updates
    ##If you use [1:] after each of the plot arguments, 
    ##You can see that the change is still positive after the first iteration
    plt.plot(ELBO_deltas.total); plt.title('Change in ELBO with each set of updates'); plt.show();

    plt.plot(ELBO_deltas.zeta_z + ELBO_deltas.zeta_x); plt.plot(ELBO_deltas.zeta_z); plt.plot(ELBO_deltas.zeta_x);
    plt.legend(['z+x', 'z', 'x'], loc='upper right'); plt.title('Change in z&x ELBO terms due to zeta update'); plt.show()

    plt.plot(ELBO_deltas.lambda_z + ELBO_deltas.lambda_z); plt.plot(ELBO_deltas.lambda_z); plt.plot(ELBO_deltas.lambda_phi);
    plt.legend(['z+phi', 'z', 'phi'], loc='upper right'); plt.title('Change in z&phi ELBO terms due to lambda update'); plt.show()

    plt.plot(ELBO_deltas.nu_mu + ELBO_deltas.nu_x); plt.plot(ELBO_deltas.nu_mu); plt.plot(ELBO_deltas.nu_x);
    plt.legend(['mu+x', 'mu', 'x'], loc='upper right'); plt.title('Change in mu&x ELBO terms due to nu update'); plt.show()

    plt.plot(ELBO_deltas.omega_mu + ELBO_deltas.omega_x); plt.plot(ELBO_deltas.omega_mu); plt.plot(ELBO_deltas.omega_x);
    plt.legend(['mu+x', 'mu', 'x'], loc='upper right'); plt.title('Change in mu&x ELBO terms due to omega update'); plt.show()

    plt.plot(ELBO_deltas.ab_tau + ELBO_deltas.ab_x); plt.plot(ELBO_deltas.ab_tau); plt.plot(ELBO_deltas.ab_x);
    plt.legend(['tau+x', 'tau', 'x'], loc='upper right'); plt.title('Change in tau&x ELBO terms due to ab update'); plt.show()

    ##Plot each datapoint with a colour corresponding to the variational cluster to which it is assigned with maximum probability
    batch_number = 0
    nonzero_datapoints = nonzero_datapoints_batches[1][np.where(nonzero_datapoints_batches[0] == batch_number)]
    inferred_zeta = inferred_latents.zeta[batch_number,nonzero_datapoints,:]
    assignments = np.argmax(inferred_zeta, axis=1)
    plt.scatter(data[batch_number,nonzero_datapoints,0],data[batch_number,nonzero_datapoints,1],c=assignments,marker='x')
    plt.gca().set_xlim([-10,10])
    plt.gca().set_ylim([-10,10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('MAP assignments of datapoints to clusters')
    plt.show()

def clusters_demo(np_seed=0, tf_seed=0, alpha=1.0, T=100, max_n_iter=20):

    data = gen_demo_data(batch_size = 1, np_seed = np_seed)
    nonzero_datapoints_batches = np.where(~np.all(data==0, axis=2))

    for n_iter in [0, 1, 2, 5, 10, max_n_iter]:
        inferred_latents = variational_inference(data, alpha=alpha, T=T, n_iter=n_iter, tf_seed=tf_seed, get_elbo=False)

        #Plot means and datapoints as points
        batch_number = 0
        nonzero_datapoints = nonzero_datapoints_batches[1][np.where(nonzero_datapoints_batches[0] == batch_number)]
        plt.scatter(data[batch_number,nonzero_datapoints,0],data[batch_number,nonzero_datapoints,1], marker='x')
        plt.scatter(inferred_latents.nu[batch_number,:,0] + 0.01*np.random.randn(T), inferred_latents.nu[batch_number,:,1],marker='o',s=30,color='r')
        
        #Plot expected standard deviation as diameter of ellipse
        patches = []; 
        diameter = 2*np.sqrt(1./np.divide(inferred_latents.a, inferred_latents.b))
        
        #Plot marginal cluster probabilities as the transparency of circle
        l1 = inferred_latents.lambda_1[batch_number,:]
        l2 = inferred_latents.lambda_2[batch_number,:]
        beta_means = np.divide(l1,l1 + l2)
        log_beta_means = np.log(beta_means + 1e-30)
        cs = np.concatenate(( [0], np.cumsum( np.log(1-beta_means+1e-30)[:-1]) )) #SBP
        beta_expectation = np.exp(log_beta_means + cs)
        beta_expectation /= (1.*np.sum(beta_expectation))               
        for k in range(T):
            circle = Ellipse((inferred_latents.nu[batch_number,k,0], inferred_latents.nu[batch_number,k,1]), diameter[batch_number,k,0], diameter[batch_number,k,1])
            plt.gca().add_artist(circle)
            circle.set_alpha(beta_expectation[k])   
        plt.gca().set_xlim([-10,10])
        plt.gca().set_ylim([-10,10])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Variational distributions at iteration ' + str(n_iter))
        plt.show()

def batch_demo(batch_size=2, np_seed=0, tf_seed=0, alpha=1.0, T=100, max_n_iter=20):

    data = gen_demo_data(batch_size = batch_size, np_seed = np_seed)
    nonzero_datapoints_batches = np.where(~np.all(data==0, axis=2))

    #Run inference
    inferred_latents = variational_inference(data, alpha=alpha, T=T, n_iter=max_n_iter, tf_seed=tf_seed, get_elbo=False)

    for batch_number in range(batch_size):
        nonzero_datapoints = nonzero_datapoints_batches[1][np.where(nonzero_datapoints_batches[0] == batch_number)]
        inferred_zeta = inferred_latents.zeta[batch_number, nonzero_datapoints, :]

        #plot weighted points 
        #https://stackoverflow.com/questions/41314736/scatterplot-wherein-each-point-color-is-a-different-mixture-of-k-colors
        HSV = [(x*1.0/T, 0.8, 0.5) for x in np.random.permutation(T)]
        RGB = np.array(map(lambda x: colorsys.hsv_to_rgb(*x), HSV))
        assignments = np.sum(np.multiply(RGB[np.newaxis, :, :], inferred_zeta[:, :, np.newaxis]),axis=1)
        plt.scatter(data[batch_number,nonzero_datapoints,0],data[batch_number,nonzero_datapoints,1],c=assignments,marker='x')
        plt.title('Weighted assignments, batch ' + str(batch_number))
        plt.gca().set_xlim([-10,10])
        plt.gca().set_ylim([-10,10])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == "__main__":
    print('Diagonal Covariance Model')
    np_seed = 23; tf_seed = 100; alpha=1.0; T=100; max_n_iter=20;
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Change in ELBO with each iteration of updates:')
    ELBO_demo(np_seed = np_seed, tf_seed = tf_seed, alpha=alpha, T=T, max_n_iter=max_n_iter)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Change in latent parameters with increasing number of updates:')
    clusters_demo(np_seed = np_seed, tf_seed = tf_seed, alpha=alpha, T=T, max_n_iter=max_n_iter)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('This script can use batches of different datasets:')
    batch_demo(batch_size = 2, np_seed = np_seed, tf_seed = tf_seed, alpha=alpha, T=T, max_n_iter=max_n_iter)

