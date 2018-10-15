import numpy as np


# Bootstrap replicate for one dimensional array

def bootstrap_replicate_1d(data, func):
    """
    Función que primero obtiene datos aleatoriamente procedente de un array. Los registros de estos datos pueden repetirse, es decir, se realizando permutaciones con repetición. Segundo calculamos algún estadístico de interés.
    
    @data: array
    @func: función del estadístico a calcular
    bs_sample: Array de la misma longitud de data pero con datos repetidos
    
    return
    
    func: estadístico que queremos calcular
    """    
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


# Generating many bootstrap replicates

def draw_bs_reps(data, func, size=1):
    """
    Función que nos guarda en una lista y posteriormente lo transforma en array, todos los estadísticos que nos calcula la función bootstrap_replicate_1d()
    """

    # Generate replicates
    bs_replicates = [bootstrap_replicate_1d(data, func) for i in range(size)]

    return np.asarray(bs_replicates)


####################################################
##  Simulación
####################################################


def compute_sampling_distribution(distribution, func, n=100, iters=1000):
    def make_sample(distribution, n=100):
        """
        Función que realiza muestreos aleatorios procedentes de una distribución específica.

        Parameters
        ----------
        distribution: Generador de números aleatorios.
        n: tamaño de la muestra de la distribución.
        """
        return distribution.rvs(n)

    def sample_stat(sample, func):
        if func == numpy.percentile:
            return numpy.percentile(sample, q=[10, 90])
        else:
            return func(sample)

    stats = [sample_stat(make_sample(distribution, n), func) for i in range(iters)]
    return numpy.array(stats)

def plot_sampling_distribution(n, xlim=None):
    """Plot the sampling distribution.
    
    n: sample size
    xlim: [xmin, xmax] range for the x axis 
    """
    sample_stats = compute_sampling_distribution(weight, numpy.mean, n, iters=1000)
    se = numpy.std(sample_stats)
    ci = numpy.percentile(sample_stats, [5, 95])
    
    pyplot.hist(sample_stats, color=COLOR2)
    pyplot.xlabel('sample statistic')
    pyplot.xlim(xlim)
    text(0.03, 0.95, 'CI [%0.2f %0.2f]' % tuple(ci))
    text(0.03, 0.85, 'SE %0.2f' % se)
    pyplot.show()
    
def text(x, y, s):
    """Plot a string at a given location in axis coordinates.
    
    x: coordinate
    y: coordinate
    s: string
    """
    ax = pyplot.gca()
    pyplot.text(x, y, s,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)