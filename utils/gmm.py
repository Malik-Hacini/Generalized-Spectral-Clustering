import numpy as np
import random

class Gaussian:
    """Multivariate Gaussian distribution for sampling."""
    
    def __init__(self, d: int, mean, cov) -> None:
        """
        Parameters:
        -----------
        d : int
            Dimension of the Gaussian
        mean : array-like
            d-dimensional mean vector
        cov : array-like
            d×d covariance matrix
        """
        self.dim=d
        self.mean=mean
        self.cov_matrix=cov
        self.points=np.array([])

    def sample(self, N: int):
        """Sample N points from the Gaussian distribution.
        
        Parameters:
        -----------
        N : int
            Number of samples
            
        Returns:
        --------
        ndarray
            Sampled points of shape (N, d)
        """
        samples=np.random.default_rng().multivariate_normal(self.mean,self.cov_matrix, N)
        points=self.points.tolist()
        points.append(samples)
        self.points=np.array(points)

        return samples


# Utility functions

def bivariate_cov_m(sigma_x, sigma_y, p):
    """Create 2D covariance matrix from standard deviations and correlation."""
    return np.array([[sigma_x**2,p*sigma_x*sigma_y],
                    [p*sigma_x*sigma_y,sigma_y**2]])

def GMM(d, N, means, covs, distrib='uniform'):
    """Sample N points from a Gaussian Mixture Model and assign labels.
    
    Parameters:
    -----------
    d : int
        Dimension of the Gaussians
    N : int
        Number of samples
    means : list
        List of d-dimensional mean vectors
    covs : list
        List of d*d covariance matrices
    distrib : str or list, optional
        'uniform' for equal probability or list of probabilities for each Gaussian
        
    Returns:
    --------
    tuple
        (samples, labels) where samples is (N, d) array and labels is (N,) array
    """
    k=len(means)
    gaussians=[]
    samples=[]
    labels=[]
    for i in range(k):
        gaussians.append(Gaussian(d,means[i],covs[i]))

    for i in range(N):
        if distrib=='uniform':
            r=np.random.default_rng().integers(k)
        else:
            r=np.random.choice(np.arange(0, k), p=distrib)
            
        samples+=list(gaussians[r].sample(1))
        labels.append(r)

    return np.array(samples), np.array(labels)

# Examples

def random_GMM(n_clusters, n_samples_fixed=False):
    if not n_samples_fixed:
        n_samples=random.randint(100,600)
    else:
        n_samples=n_samples_fixed

    means=[]
    for i in range(n_clusters):
        means.append([random.uniform(-10,10),random.uniform(-10,10)])
    covs_initial=[np.add(np.random.default_rng().random(size=(2,2))*(3-0.2), 0.2*np.ones((2,2)))  for i in range(n_clusters)]
    covs_symmetry=[np.matmul(cov,np.transpose(cov)) for cov in covs_initial]
    data, labels=GMM(2,n_samples,means,covs_symmetry)    
    return data, labels
    

def interesting_gmm():
    means=[[0,0],[3,2],[-4,-2],[-2,5]]
    covs=[1*np.identity(2),1*np.identity(2),1*np.identity(2),1*np.identity(2)]
    data, labels=GMM(2,350,means,covs)    
    return data, labels

def circular_GMM(n_clusters, n_samples, sigma, factor):
    means=[(i*factor*sigma,i*factor*sigma) for i in range(n_clusters)]
    covs=[sigma*np.identity(2) for i in range(n_clusters)]
    return GMM(2,n_samples,means,covs)