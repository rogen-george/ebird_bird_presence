from scipy.stats import dirichlet
from scipy.special import digamma
from scipy.special import polygamma
import numpy as np
import math
from numpy import errstate,isneginf,array

def estimate_precision( data ):
    # Fix the mean
    mean = [ 1 / len(data[0]) ] * len(data[0])

    dimensions = len(mean)
    N = len( data )
    p_k = np.zeros(dimensions)

    # Number of iterations for Newton Update
    iterations = 100

    # Calculate the value of log p_k
    for dimension in range( dimensions ):
        with errstate(divide='ignore'):
            p_k[dimension] = np.sum ( np.log( data[:, dimension] ) ) / N
        p_k[isneginf(p_k)]=0

    denominator = 0
    for i, m_k in enumerate(mean):
        denominator += m_k * ( p_k[i] - np.log(m_k) )

    # Calculate the initial value of precision s
    numerator = (dimensions - 1) / 2
    initial_s = - 1* numerator / denominator

    s = initial_s
    for i in range ( iterations ):
        #print (s)
        second_term = 0
        third_term = 0
        d2_term = 0
        for i, m_k in enumerate(mean):
            second_term += m_k * digamma( s * m_k )
            third_term += m_k * p_k[i]
            d2_term += ( (m_k) ** 2 ) * polygamma(1, s * m_k)

        d_log_likelihood =   N * ( digamma(s) - second_term + third_term )
        d2_log_likelihood =  N * ( polygamma(1, s) - d2_term )

        # Update the value of s after each iteration
        s = 1 / ( (1 / s) + (1 / (s ** 2) ) *  ( ( 1 / d2_log_likelihood ) * d_log_likelihood ) )

    return (s)

def balanced_rearrangement_matrices(n, k):
    n_n = n
    k_k = k

    #N_K = n_n/k_k

    x = np.zeros((k , n))
    matrices = []

    while ( len(matrices) < 500 ):
        n = n_n
        k = k_k
        x = np.zeros((k_k , n_n))
        #n_by_k = n_n / k_k

        i = 0

        for i in range(n):
            # First row - First col
            #if ( i < k_k - 1):
                #n_by_k = N_K
            x[:,i] = np.random.random(k)
                # n_by_k = ( n_by_k - np.sum( x[i, 0:i] ) )
            x[:,i]  =  ( x[:,i]  / np.sum( x[:,i] ) )

                # First column
                # x[(i + 1):,i] = np.random.random(k - 1)
                # t = ( 1 - np.sum( x[0:(i + 1),i] ) )
                # x[(i + 1):,i] = ( x[(i + 1):,i] / np.sum( x[(i + 1):,i] ) ) * t

                # n -= 1
                #k -= 1

        matrices.append(x)
    return (matrices)
