#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:30:19 2024

@author: prcohen
"""

import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=52355) 
import time

class Regressions ():
    ''' 
    This class runs all pairwise regressions given a design matrix X.  The regressions
    and much else are derived from the covariance matrix np.cov(X).  
    '''
    
    def __init__(self,X,labels=None,setup=True,get_residuals=True,get_y_hat=True):
        self.X = X
        self.labels=labels
        
        if setup: 
            # number of rows and columns in X
            self.N,self.M = self.X.shape
            # indices of the pairwise regressions
            self.pairs = np.array([[i,j] for i in range(self.M) for j in range(self.M) if i != j])
            # variance-covariance matrix
            self.C = np.cov(self.X, rowvar=False) 
            # variances of the N variables in X
            self.V = self.C.diagonal()
            # run all pairwise regressions to get the M*M matrices of regression coefficients b0 and b1
            self.all_regressions()
            # run SSres to get the M*M matrices of R^2 and sum of squared residuals
            if get_residuals: self.calc_SSres()
            
            # Get the matrix of predictions y_hat.
            # This can be very slow because it considers every data point (N*M) 
            # and all M * M regression models. 
            if get_y_hat: self.y_hat = self.calc_y_hat()
        
        
    def all_regressions (self,report=False):
        ''' This regresses every column in X on every other:  x_i = b0 + x_j * b1.  
        It produces two square M x M matrices:  the matrix of slopes, b1; and the 
        matrix of intercepts, b0.  Note that the diagonals of these matrices are 
        uninformative because they are regressions of a variable on itself. '''
        
        t = time.time()
        
        # The slope of the regression of x_j on x_i is cov(x_i,x_j)/var(x_i)
        self.b1 = self.C / self.V[:, np.newaxis]
        
        # Since the regression line goes through the means of x and y, we can
        # solve for the intercept:  b0 = mean(Y) - b1 * mean(X). It's vectorized:
        
        means = np.mean(self.X, axis=0)
        self.b0 = means - self.b1 * means[:, np.newaxis]
        
        if report: print(f"all_regressions took {time.time()-t:.5f} secs.")
        
    def calc_SSres (self,report=False):
        ''' This returns the M*M matrix of SSres for all pairwise regressions
        and also the M*M matrix of R^2.  It does not return the residuals 
        themselves, nor can it, because it solves for SSres given the variance-
        covariance matrix. This is done for speed. To get the residuals, themselves, 
        use the calc_residuals method. 
        '''
        self.R2 = (self.C**2) / (self.V[:, np.newaxis] * self.V)
        self.SSres = (1 - self.R2) * self.V * (self.N-1)
        if report: print(f"Calculating SSres and R^2 took {time.time()-t:.4f} seconds")
        
    # def calc_y_hat (self,X=None,report=False):
    #     ''' This calculates y_hat for all pairwise regressions simultaneously.
    #     To get y_hat for the i,j regression: self.y_hat[:,i,i,j]. 
    #     '''
    #     t = time.time()
    #     n,m = self.N,self.M
    #     x = self.X.reshape(n*m)
    #     B1 = self.b1.reshape(m*m,1)
    #     B0 = self.b0.reshape(m*m,1) 
    #     self.y_hat = (B1 * x + B0).T.reshape(n,m,m,m)
    #     if report: print(f"Calculating y_hat took {time.time()-t:.4f} seconds")
        
    def calc_y_hat (self,b0=None,b1=None,report=False):
         ''' This calculates y_hat for all pairwise regressions.
         To get y_hat for the i,j regression: self.y_hat[:,i,i,j]. If X is None,
         then we assume the design matrix is self.X, the matrix for which we 
         calculated our regression coefficient matrices self.b0 and self.b1.  
         But we could also calculate y_hat for a different set of regression 
         coefficients, provided M is the same for both. This is done to compare
         predictions for self.X based on the models for self.X with predictions
         based on a different set of models; for example, if self.X represents
         controls, we might want to compare predictions based on controls-derived
         models with predictions based on case-derived models. 
         
         M is the number of columns in X, but in addition we assume that they 
         are the same columns, i.e., the same biomarkers, 
         '''
         t = time.time()
         _b0 = b0 if b0 is not None else self.b0
         _b1 = b1 if b1 is not None else self.b1
         n,m = self.X.shape
         x = self.X.reshape(n*m)
         B1 = _b1.reshape(m*m,1)
         B0 = _b0.reshape(m*m,1) 
         y_hat = (B1 * x + B0).T.reshape(n,m,m,m)
         if report: print(f"Calculating y_hat took {time.time()-t:.4f} seconds")
         return y_hat
        
        
    def calc_residuals (self,i,j,squared=True,y_hat=None,report=False):
        ''' This calculates residuals for the i,j regression.  It does not calculate
        residuals for all regressions. This follows the convention 
        that x = self.X[:,i] and y = self.X.[:,j] and y_hat from the regression 
        is self.y_hat[:,i,i,j]. By default the residuals are squared. This can 
        calculate residuals between self.X and any y_hat, not necessarily the y_hat
        for self.X.  Thus we can calculate y_hat for a different X (e.g., for
        cases if self.X is noncases, or vice versa, and test whether the residuals
        are larger when we use the regression models for the "wrong X" to calculate
        y_hat.  
        '''
        t = time.time()
        # _y_hat is by default self.y_hat but could be any other y_hat that has
        # the same shape as self.X.
        _y_hat = y_hat if y_hat is not None else self.y_hat
        errs = self.X[:,j] - _y_hat[:,i,i,j]
        if report: print(f"Calculating y_hat took {time.time()-t:.4f} seconds")
        return errs**2 if squared else errs
    
    def plot_regression (self,i,j,c):
        x,y = self.X[:,i],self.X[:,j]
        plt.scatter(x,y,s=.5,c = c)
        plt.plot(x,self.b1[i,j]*x + self.b0[i,j],c=c,linewidth=.25)
        
    def plot_pair (self,x0,y0,x1,y1):
        m0,b0 = plot_regression(x0,y0,'green')
        m1,b1 = plot_regression(x1,y1,'darkorange')
        x = np.arange(min(min(x0),min(x1)),max(max(x0),max(x1)))
        midline_m, midline_b = (m0+m1)/2,(b0+b1)/2
        plt.plot(x,x*midline_m + midline_b,linewidth=.5,c='grey')