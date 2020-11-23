from statistics import median
import random
import numpy as np
from sklearn.preprocessing import normalize
from numpy import errstate,isneginf,array
from dca import estimate_precision, balanced_rearrangement_matrices

# Takes data and a target dimension
# To Do : Change the number of iterations and return
def genetic_algorithm( data , k ):

    n = len( data[0] )
    iterations = 10

    matrix_population = balanced_rearrangement_matrices(n, k)
    #print ( matrix_population )
    for i in range(iterations):
        #print ("Iteration ------------------------- ", i )
        dirichilet_correlation = []

        for matrix in matrix_population:
            #print ( np.sum( matrix[:,0] ) )
            reduced_data = np.matmul( matrix , data.transpose() ).transpose()
            new_matrix = normalize( reduced_data, norm = 'l1', axis = 0 )
            dirichilet_correlation.append( estimate_precision( reduced_data ) )

        updated_population = []
        min_correlation = dirichilet_correlation.index( min(dirichilet_correlation) )
        # Add the matrix with the min population

        if not np.all( matrix_population[min_correlation] == 0 ):
            updated_population.append( matrix_population[min_correlation] )

        median_dc = median(dirichilet_correlation)
        if median_dc == 0 :
            return prev
        dc_updated = np.array ( list( ( map( lambda x : min( x / median_dc, 1 ), dirichilet_correlation) ) ) )
        with errstate(divide='ignore'):
            fitness = - 1 * np.log( dc_updated )
        fitness[isneginf(fitness)] = 0
        fitness = np.nan_to_num(fitness)

        # sort the BR matrices based on fitness
        # Arg sort gives the indices of the sorted matrix in order. Multiply -1 to get decreasing order.
        fitness_sorted = np.argsort(-1 * fitness)
        # Take the first half to be added to the new population
        # Add to new population if fitness > 0
        for matrix in (fitness_sorted[0:len(fitness_sorted) // 2]):
            if fitness[matrix] > 0 and not np.all( matrix_population[matrix] == 0 ):
                updated_population.append(matrix_population[matrix])

        # Mutation operation where two random matrices are choosen weighed by their fitness
        # and used to create a new matrix which is added to the Population
        for _ in range( len(matrix_population) // 2 ):
            parent1 = random.choice(range(len(matrix_population)))
            parent2 = random.choice(range(len(matrix_population)))
            child = ( matrix_population[parent1] * fitness[parent1] ) + ( matrix_population[parent2] * fitness[parent2] )
            child = normalize( child, norm = 'l1', axis = 0)
            #print ( "child" , child )
            if not np.all( child == 0 ):
                updated_population.append( child )
        #print ("Updated Population ", updated_population)

        prev = matrix_population
        matrix_population = updated_population
        if not any(fitness):
            return prev

        if not matrix_population:
            return prev
        return matrix_population
