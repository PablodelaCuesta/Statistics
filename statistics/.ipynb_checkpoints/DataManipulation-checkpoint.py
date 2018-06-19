"""
Funciones para el manejo de DataFrames.
"""

# important libraries
import numpy as np
import pandas as pd

# Función que realiza variaciones con repetición

def SampleRows(df, nrows, replace=False):
    """
    function that takes random samples from a DataFrame.
    
    Args:
        df(DataFrame): DataFrame
        nrows (int): number of rows
        replace (boolean): if we want repeat values or not.
        
    Returns:
        Returns a dataframe with the numbers of rows indicate.
    """
    # select random indices
    indices = np.random.choice(df.index, size=nrows, replace=replace)
    
    # sample
    sample = df.loc[indices]
    
    return sample