from numpy.typing import NDArray
import numpy as np

def cohen_d(data1 : NDArray,
            data2 : NDArray) -> tuple[int, int]:
    """A function that return the cohen's D and the direction of the difference

    Args    
        data1 : An array with the first set of data
        data2 : An array with the second set of data

    Return:
        D : The strenght of the difference
        sense : 1 if data1 is superior, -1 else 
    """

    mean1 = data1.mean()
    mean2 = data2.mean()

    n1, n2 = len(data1), len(data2)

    s1, s2 = data1.std(ddof=1), data2.std(ddof=1)

    s = ((s1**2)*(n1-1)) + ((s2**2)*(n2-1))
    s = s / (n1+n2-2)
    s = np.sqrt(s) 

    D = (mean1-mean2)/s
    # if mean1 == mean2 it is still -1, but we don't expect it to happen, also, if it happens, the D will be 0
    sense = 1 if mean1 > mean2 else -1

    return (D, sense)

def cliff_delta(data1 : NDArray,
                data2 : NDArray) -> tuple[int, int]:
    """Version of cohen's D if the data are not normally distributed
 
    Args    
        data1 : An array with the first set of data
        data2 : An array with the second set of data

    Return:
        D : The strenght of the difference
        sense : 1 if data1 is superior, -1 else 
    """

    # Here we measur the superiority of the distribution 1 over the distribution 2
    superiority = lambda dist1, dist2 : np.sum([sum(d1 > dist2) for d1 in dist1])
    sup1, sup2 = optimized_superiority(dist1=data1, dist2=data2), optimized_superiority(dist1=data2, dist2=data1)
    sample_size = len(data1)*len(data2)

    delta = (sup1-sup2)/sample_size
    sense = 1 if sup1 > sup2 else -1

    return (delta, sense)

def optimized_superiority(dist1, dist2):
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    comparison = dist1[:, None] > dist2  # shape (len(dist1), len(dist2))
    return np.sum(comparison)