# Librerías a importar
import numpy as np

def ecdf(data):
    """Compute ECDF for a one dimensional array of measurements."""
    
    # Number of points
    n = len(data)
    
    # x-data for the ECDF
    x = np.sort(data)
    
    # y-data for the ECDF
    y = np.arange(1, n + 1) / n
    
    return x, y