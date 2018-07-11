#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:07:07 2018

@author: pablo
"""

# Librerías a importar
import numpy as np


"""
Funciones estadísticas importantes
"""


# The next function computes ECDF

def ecdf(data):
    """Compute ECDF for a one dimensional array of measurements."""
    
    # Number of points
    n = len(data)
    
    # x-data for the ECDF
    x = np.sort(data)
    
    # y-data for the ECDF
    y = np.arange(1, n + 1) / n
    
    return x, y


# Bootstrap replicate for one dimensional array

def bootstrap_replicate_1d(data, func):
    """ Compute a single value of a statistc. """    
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


# Generating many bootstrap replicates

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Generate replicates
    bs_replicates = [bootstrap_replicate_1d(data, func) for i in range(size)]

    return bs_replicates
    

# A function to do pairs bootstrap


def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps


# Para conocer el número de 'bins' utilizamos la regla de Freedman-Diaconis

def freedman_diaconis(data):
    """Devuelve el número de óptimo de columnas de un histograma."""
    # Calculamos los percentiles 25 y 75 para conocer el IQR
    percentile25, percentile75 = np.percentile(data, [25, 75])
    
    # Realizamos la diferencia y obtenemos el IQR
    IQR = percentile75 - percentile25

    # Calculamos el número h
    h = 2 * IQR * len(data) ** (-1/3)
    
    # finalmente
    number_bins = int(np.round((np.max(data) - np.min(data)) / h))
    
    return number_bins

def num_bins(data):
    """
    Devuele el número óptimo de columnas para un histograma.
    """
    # longuitud del dataset
    long = len(data)
    
    #calculo del número de bins
    n_bins = int(np.sqrt(long))
    
    return n_bins


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

def draw_perm_reps1d(data_1, data_2, size=1):
    """
    Realiza permutaciones sin repetición de un set de datos.
    """
    perm_coef = np.empty(size)
    
    for i in range(size):
        #Permutamos los valores
        perm_data = np.random.permutation(data_1)
        
        # Coeficiente de Pearson
        perm_coef[i] = pearson_r(perm_data, data_2)
        
    return perm_coef
