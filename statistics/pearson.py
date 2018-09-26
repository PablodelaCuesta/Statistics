import numpy as np

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

def draw_perm_reps1d(data_perm, data_2, size=1):
    """
    Nos realiza permutaciones sin repetición de un set de datos.
    """
    perm_coef = np.empty(size)
    
    for i in range(size):
        #Permutamos los valores
        perm_data = np.random.permutation(data_perm)
        
        # Coeficiente de Pearson
        perm_coef[i] = pearson_r(perm_data, data_2)
        
    return perm_coef