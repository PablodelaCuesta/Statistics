"""
Funciones de apoyo para la relación de dos variables
"""
import numpy as np

def cov(xs, ys, meanx=None, meany=None):
    """
    cov computes deviations from the sample means, or you can provide known means.    
    """
    
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    covariance = np.dot(xs - meanx, ys - meany) / len(xs)
    
    return covariance


def correlation(xs, ys):
    """
    Computes the correlation coeficient.
    """
    
    # define arrays
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    # correlation coeficient definition
    corr = cov(xs, ys) / np.sqrt(np.var(xs) * np.var(ys))
    
    return corr


def Spearmancorr(xs, ys):
    """
    Computes the spearman correlation coeficient.
    
    xs: Pandas Series
    ys: Pandas Series
    """
    
    xranks = xs.rank()
    yranks = ys.rank()
    
    corr = correlation(xranks, yranks)
    
    return corr