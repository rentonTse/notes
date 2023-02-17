#!/usr/bin/env python

"""
Weighted Principal Component Analysis using Expectation Maximization

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

Returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...
    
Stephen Bailey, Spring 2012
"""

from __future__ import division, print_function
from matplotlib.font_manager import weight_dict

import random
import numpy as np
import sys
from scipy.sparse import dia_matrix
import scipy.sparse.linalg
import math

class Model(object):
    """
    A wrapper class for storing data, eigenvectors, and coefficients.
    
    Returned by empca() function.  Useful member variables:
      Inputs: 
        - eigvec [nvec, nvar]
        - data   [nobs, nvar]
        - weights[nobs, nvar]
      
      Calculated from those inputs:
        - coeff  [nobs, nvec] - coeffs to reconstruct data using eigvec
        - model  [nobs, nvar] - reconstruction of data using eigvec,coeff
    
    Not yet implemented: eigenvalues, mean subtraction/bookkeeping
    """
    def __init__(self, eigvec, data, weights):
        """
        Create a Model object with eigenvectors, data, and weights.
        
        Dimensions:
          - eigvec [nvec, nvar]  = [k, j]
          - data   [nobs, nvar]  = [i, j]
          - weights[nobs, nvar]  = [i, j]
          - coeff  [nobs, nvec]  = [i, k]        
        """
        self.eigvec = eigvec
        self.nvec = eigvec.shape[0]
        
        self.set_data(data, weights)

        
    def set_data(self, data, weights):
        """
        Assign a new data[nobs,nvar] and weights[nobs,nvar] to use with
        the existing eigenvectors.  Recalculates the coefficients and
        model fit.
        """
        self.data = data
        self.weights = weights

        self.nobs = data.shape[0]
        self.nvar = data.shape[1]
        self.coeff = np.zeros( (self.nobs, self.nvec) )
        self.model = np.zeros( self.data.shape )
        
        #- Calculate degrees of freedom
        ii = np.where(self.weights>0)
        self.dof = self.data[ii].size - self.eigvec.size  - self.nvec*self.nobs
        
        #- Cache variance of unmasked data
        self._unmasked = ii
        self._unmasked_data_var = np.var(self.data[ii])
        
        self.solve_coeffs()
        
    def solve_coeffs(self):
        """
        Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]
        """
        for i in range(self.nobs):
            #- Only do weighted solution if really necessary
            if np.any(self.weights[i] != self.weights[i,0]):
                self.coeff[i] = _solve(self.eigvec.T, self.data[i], self.weights[i])
            else:
                self.coeff[i] = np.dot(self.eigvec, self.data[i])
            
        self.solve_model()
            
    def solve_eigenvectors(self, smooth=None):
        """
        Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]
        """

        #- Utility function; faster than numpy.linalg.norm()
        def norm(x):
            return np.sqrt(np.dot(x, x))
            
        #- Make copy of data so we can modify it
        data = self.data.copy()

        #- Solve the eigenvectors one by one
        for k in range(self.nvec):

            #- Can we compact this loop into numpy matrix algebra?
            c = self.coeff[:, k]
            for j in range(self.nvar):
                w = self.weights[:, j]
                x = data[:, j]
                # self.eigvec[k, j] = c.dot(w*x) / c.dot(w*c)
                # self.eigvec[k, j] = w.dot(c*x) / w.dot(c*c)
                cw = c*w
                self.eigvec[k, j] = x.dot(cw) / c.dot(cw)
                                                
            if smooth is not None:
                self.eigvec[k] = smooth(self.eigvec[k])

            #- Remove this vector from the data before continuing with next
            #? Alternate: Resolve for coefficients before subtracting?
            #- Loop replaced with equivalent np.outer(c,v) call (faster)
            # for i in range(self.nobs):
            #     data[i] -= self.coeff[i,k] * self.eigvec[k]
                                
            data -= np.outer(self.coeff[:,k], self.eigvec[k])    

        #- Renormalize and re-orthogonalize the answer
        self.eigvec[0] /= norm(self.eigvec[0])
        for k in range(1, self.nvec):
            for kx in range(0, k):
                c = np.dot(self.eigvec[k], self.eigvec[kx])
                self.eigvec[k] -=  c * self.eigvec[kx]
                    
            self.eigvec[k] /= norm(self.eigvec[k])
        
        #- Recalculate model
        self.solve_model()
           
    def solve_model(self):
        """
        Uses eigenvectors and coefficients to model data
        """
        for i in range(self.nobs):
            self.model[i] = self.eigvec.T.dot(self.coeff[i])
                       
    def chi2(self):
        """
        Returns sum( (model-data)^2 / weights )
        """
        delta = (self.model - self.data) * np.sqrt(self.weights)
        return np.sum(delta**2)
        
    def rchi2(self):
        """
        Returns reduced chi2 = chi2/dof
        """
        return self.chi2() / self.dof
        
    def _model_vec(self, i):
        """Return the model using just eigvec i"""
        return np.outer(self.coeff[:, i], self.eigvec[i])
        
    def R2vec(self, i):
        """
        Return fraction of data variance which is explained by vector i.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """
        
        d = self._model_vec(i) - self.data
        return 1.0 - np.var(d[self._unmasked]) / self._unmasked_data_var
        
    def R2(self, nvec=None):
        """
        Return fraction of data variance which is explained by the first
        nvec vectors.  Default is R2 for all vectors.
        
        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """
        if nvec is None:
            mx = self.model
        else:            
            mx = np.zeros(self.data.shape)
            for i in range(nvec):
                mx += self._model_vec(i)
            
        d = mx - self.data

        #- Only consider R2 for unmasked data
        return 1.0 - np.var(d[self._unmasked]) / self._unmasked_data_var
                
def _random_orthonormal(nvec, nvar, seed=1):
    """
    Return array of random orthonormal vectors A[nvec, nvar] 

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        np.random.seed(seed)
        
    A = np.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= np.dot(A[j], A[i]) * A[j]
            A[i] /= np.linalg.norm(A[i])

    return A

def _solve(A, b, w):
    """
    Solve Ax = b with weights w; return x
    
    A : 2D array
    b : 1D array length A.shape[0]
    w : 1D array same length as b
    """
  
    #- Apply weights
    nvar = len(w)
    W = dia_matrix((w, 0), shape=(nvar, nvar))
    bx = A.T.dot( W.dot(b) )
    Ax = A.T.dot( W.dot(A) )
    
    b = A.T.dot( w*b )
    A = A.T.dot( (A.T * w).T )

    if isinstance(A, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        # x = np.linalg.solve(A, b)
        x = np.linalg.lstsq(A, b)[0]
        
    return x

    
#-------------------------------------------------------------------------

def empca(data, weights=None, niter=25, nvec=5, smooth=0, randseed=1, silent=False):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights
    
    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]
      
    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize
    
    Returns Model object
    """

    if weights is None:
        weights = np.ones(data.shape)

    if smooth>0:
        smooth = SavitzkyGolay(width=smooth)
    else:
        smooth = None

    #- Basic dimensions
    nobs, nvar = data.shape
    assert data.shape == weights.shape

    #- degrees of freedom for reduced chi2
    ii = np.where(weights > 0)
    dof = data[ii].size - nvec*nvar - nvec*nobs 

    #- Starting random guess
    eigvec = _random_orthonormal(nvec, nvar, seed=randseed)
    
    model = Model(eigvec, data, weights)
    model.solve_coeffs()
    
    if not silent:
        print("       iter        R2             rchi2")
    
    for k in range(niter):
        model.solve_coeffs()
        model.solve_eigenvectors(smooth=smooth)
        if not silent:
            print('EMPCA %2d/%2d  %15.8f %15.8f' % \
                (k+1, niter, model.R2(), model.rchi2()))
            sys.stdout.flush()

    #- One last time with latest coefficients
    model.solve_coeffs()

    if not silent:
        print("R2:", model.R2())
    
    return model

def classic_pca(data, nvec=None):
    """
    Perform classic SVD-based PCA of the data[obs, var].
    
    Returns Model object
    """
    u, s, v = np.linalg.svd(data)
    if nvec is None:
        m = Model(v, data, np.ones(data.shape))    
    else:
        m = Model(v[0:nvec], data, np.ones(data.shape))
    return m


class SavitzkyGolay(object):
    """
    Utility class for performing Savitzky Golay smoothing
    
    Code adapted from http://public.procoders.net/sg_filter/sg_filter.py
    """
    def __init__(self, width, pol_degree=3, diff_order=0):
        self._width = width
        self._pol_degree = pol_degree
        self._diff_order = diff_order
        self._coeff = self._calc_coeff(width//2, pol_degree, diff_order) 

    def _calc_coeff(self, num_points, pol_degree, diff_order=0):
    
        """
        Calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf
    
        num_points   means that 2*num_points+1 values contribute to the
                     smoother.
    
        pol_degree   is degree of fitting polynomial
    
        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first 
                                                 derivative of function.
                     and so on ...
        """
    
        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....
    
        x = np.arange(-num_points, num_points+1, dtype=int)
        monom = lambda x, deg : math.pow(x, deg)
    
        A = np.zeros((2*num_points+1, pol_degree+1), float)
        for i in range(2*num_points+1):
            for j in range(pol_degree+1):
                A[i,j] = monom(x[i], j)
            
        # calculate diff_order-th row of inv(A^T A)
        ATA = np.dot(A.transpose(), A)
        rhs = np.zeros((pol_degree+1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = np.linalg.solve(ATA, rhs)
    
        # calculate filter-coefficients
        coeff = np.dot(A, wvec)
    
        return coeff
    
    def __call__(self, signal):
        """
        Applies Savitsky-Golay filtering
        """
        n = np.size(self._coeff-1)//2
        res = np.convolve(signal, self._coeff)
        return res[n:-n]


def _main():
    np.random.seed(1)
    nobs = 10
    nvar = 10
    nvec = 1
    data = np.zeros(shape=(nobs, nvar))

    #- Generate data
    x = np.linspace(0, 2*np.pi, nvar)
    for i in range(nobs):
        for k in range(nvec):
            c = np.random.normal()
            data[i] += 5.0*nvec//(k+1)**2 * c * np.sin(x*(k+1))

    #- Add noise
    sigma = np.ones(shape=data.shape)
    for i in range(nobs//10):
        sigma[i] *= 5
        sigma[i, 0:nvar//4] *= 5

    weights = 1.0 / sigma**2    
    noisy_data = data + np.random.normal(scale=sigma)


    # Gaussian noisy
   
    # sigma = 10

    # noisy_data = data + random.gauss(0,sigma)
    # weights = np.ones(shape=data.shape)
    

    # print(noisy_data)

    # for i in range(weights.shape[0]):
    #     for j in range(weights.shape[1]):
    #         if noisy_data[i][j] <= 2*sigma :
    #             weights[i][j] = sigma
    #         else:
    #             weights[i][j] = np.sqrt(sigma**2 + noisy_data[i][j])
    # weights = 1.0 / weights**2            
    # print(weights)                 
 
    print("Data is ",noisy_data)


    print("Testing empca")
    m0 = empca(noisy_data, weights, niter=20)
    print(m0.eigvec.shape,m0.data.shape,m0.coeff.shape)
    print(m0.coeff[0])
    
    
    print("Testing classic PCA")
    m2 = classic_pca(noisy_data)
  
    print("R2", m2.R2())
    
    try:
        import pylab as P
    except ImportError:
        print >> sys.stderr, "pylab not installed; not making plots"
        sys.exit(0)
        
    # P.subplot(311)
    # # P.imshow(noisy_data)
    # # P.plot(m0.coeff[0])
    # P.plot(m0.eigvec[0])
    # # P.imshow(m0.eigvec,cmap='gray')
    # # P.ylim(-0.2, 0.2)
    # P.ylabel("EMPCA")
    # P.title("Eigenvectors")
    
    # P.subplot(312)
    # for i in range(nvec): P.plot(m0.eigvec[i])
    # P.ylim(-0.2, 0.2)
    # P.ylabel("EMPCA")
    
    
    # P.subplot(313)
    # for i in range(nvec): P.plot(m2.eigvec[i])
    # P.ylim(-0.2, 0.2)
    # P.ylabel("Classic PCA")
    # P.show()

    # from mpl_toolkits import mplot3d
    # X = np.array([m0.coeff[:,0]])
    # Y = np.array([m0.eigvec[0]])
    # z = m0._model_vec(0) 

    # fig = P.figure()
    # ax = P.axes(projection='3d')
  
    # ax.plot_surface(X,Y,z,cmap='viridis', edgecolor='none')
    # ax.set_title('Surface plot')
    # P.show()
    # P.imshow(z)
    # print(X.T.shape,Y.shape)     
        
if __name__ == '__main__':
    _main()
    






    

