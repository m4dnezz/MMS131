import numpy as np
import random
from tqdm.auto import tqdm
import itertools
import multiprocessing
import psutil
import os
import time


############################################
# Made by Niclas Persson 2023/04/26
###########################################

# Change these parameters as desired
filename = "data_ga.txt"
populationSize = 200
numberOfGenerations = 5000
# Do not change these
numberOfParameters = 6
maximumParameterValue = 2

#############################


def importdata(file):
    alldata = np.genfromtxt(file)
    xdata = alldata[:, 0]
    ydata = alldata[:, 1]
    gdata = alldata[:, 2]
    return xdata, ydata, gdata, alldata


def InitializePopulation(populationSize=100, numberOfParameters=6, empty=False):
    if not empty:
        pop = np.random.uniform(-maximumParameterValue, maximumParameterValue + 0.00001,
                                size=(populationSize, numberOfParameters))
    else:
        pop = np.zeros(shape=(populationSize, numberOfParameters))
    return pop


def EvaluateIndividual(ind, functionData, xdata, ydata, gdata):
    alldata = functionData
    x = xdata
    y = ydata
    g = gdata
    approx = (1 + ind[0] * x + ind[1] * x ** 2 + ind[2] * x ** 3) / (1 + ind[3] * y + ind[4] * y ** 2 + ind[5] * y ** 3)
    error = np.sqrt(np.sum((approx - g) ** 2) * (1 / populationSize))
    fitness = np.exp(-error)
    return fitness


def TournamentSelect(fitness, tournamentProbability, size=2):
    tp = tournamentProbability
    random_1 = random.randint(0, len(fitness) - 1)
    random_2 = random.randint(0, len(fitness) - 1)
    # Make sure that we do not have the same value

    while random_1 == random_2:
        random_1 = random.randint(0, len(fitness) - 1)
        random_2 = random.randint(0, len(fitness) - 1)

    # Get the individual from the population
    ind_1 = fitness[random_1]
    ind_2 = fitness[random_2]

    if random.random() < tp:
        if ind_1 < ind_2:
            ind = random_2
        else:
            ind = random_1
    else:
        if ind_1 < ind_2:
            ind = random_1
        else:
            ind = random_2
    return ind


def Cross(chromosome_1, chromosome_2, p_cross):
    if random.random() > p_cross:
        return chromosome_1, chromosome_2
    else:
        cut = random.randint(0, 6)
        new_ind_1 = np.concatenate([chromosome_1[0:cut], chromosome_2[cut:, ]])
        new_ind_2 = np.concatenate([chromosome_2[0:cut], chromosome_1[cut:, ]])
        return new_ind_1, new_ind_2


def Mutate(originalchromosome, mutationprobability, creepprobability, creeprate):
    newchromosome = []
    for gene in originalchromosome:
        # Do not perform any mutation
        if random.random() > mutationprobability:
            newchromosome.append(gene)
        # Perform Mutation
        else:
            # Perform Full-range mutation
            if random.random() > creepprobability:
                newchromosome.append(np.random.uniform(-2, 2))
            # Perform Creep mutation
            else:
                mutation = random.uniform(-creeprate / 2, creeprate / 2)
                newchromosome.append(gene + mutation)
    return newchromosome


def main(mp, cp, tp, cr, crp):
    mutationProbability = mp
    crossoverProbability = cp
    tournamentProbability = tp
    creepRate = cr
    creepProbability = crp

    # Initialize everything
    xdata, ydata, gdata, alldata = importdata(filename)
    population = InitializePopulation(populationSize)
    igeneration = 0
    maximum_fitness = []
    best_chromosome = None
    x = []

    # Run the evolution
    while igeneration < numberOfGenerations:
        # Reset parameters for each population
        fitness_population = []
        best_index = None
        fitness = 0

        # Evaluate generation
        for i in range(len(population)):
            f_ind = EvaluateIndividual(population[i], alldata, xdata, ydata, gdata)
            fitness_population.append(f_ind)
            if f_ind > fitness:
                fitness = f_ind
                best_index = i

        maximum_fitness.append(fitness)

        # Save best chromosome
        best_chromosome = population[best_index]

        # Generate temporary population
        temp_pop = InitializePopulation(populationSize, empty=True)

        # Transfer from old population to temporary
        for m in range(0, populationSize, 2):
            # Select two individuals based tournament selection
            ind_1 = TournamentSelect(fitness_population, tournamentProbability)
            ind_2 = TournamentSelect(fitness_population, tournamentProbability)

            # Remove them from population
            np.delete(fitness_population, ind_1)
            np.delete(fitness_population, ind_2)

            # Extract chromosomes from index
            chromosome_1 = population[ind_1]
            chromosome_2 = population[ind_2]

            # Perform crossover
            new_individual_pair = Cross(chromosome_1, chromosome_2, crossoverProbability)
            temp_pop[m] = new_individual_pair[0]
            temp_pop[m + 1] = new_individual_pair[1]

        # Mutate temp poulation
        for n in range(1, populationSize):
            original_chromosome = temp_pop[n]
            mutated_chromosome = Mutate(original_chromosome, mutationProbability, creepProbability, creepRate)
            temp_pop[n] = mutated_chromosome

        temp_pop[0] = best_chromosome  # Elitism
        population = temp_pop
        x.append(igeneration)
        igeneration += 1

    with open("results.txt", "a") as file:
        file.write(f"Fitness: {fitness}, Pop_size: {populationSize}, Generations: {numberOfGenerations}\n"
                   f"TP: {tournamentProbability}, COP: {crossoverProbability} MP: {mutationProbability},"
                   f"CP: {creepProbability}, CR: {creepRate}\n"
                   f"Chromosome: {best_chromosome} \n \n")

    return fitness


def evaluate_params(param):
    mp, cp, tp, cr, crp = param
    result = main(mp, cp, tp, cr, crp)
    return param, result

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.IDLE_PRIORITY_CLASS)

if __name__ == "__main__":
    mp_test = [0.05, 0.1, 0.15, 0.2]
    cp_test = [0.4, 0.5, 0.6, 0.7, 0.8]
    tp_test = [0.7, 0.75, 0.80, 0.85]
    cr_test = [0.01, 0.02, 0.03, 0.04, 0.05]
    crp_test = [0.2, 0.3, 0.5, 0.6, 0.7]

    for knulla in range(4):
        # Create a pool of processes
        pool = multiprocessing.Pool(None, limit_cpu)

        # Evaluate the parameters in parallel
        results = list(
            tqdm(pool.imap_unordered(evaluate_params, itertools.product(mp_test, cp_test, tp_test, cr_test, crp_test)),
                 total=len(mp_test) * len(cp_test) * len(tp_test) * len(cr_test) * len(crp_test), position=0, leave=True,
                 colour="green"))

        # Find the best parameters and result
        best_param = None
        best_result = float('-inf')
        for param, result in results:
            if result > best_result:
                best_param = param
                best_result = result

        with open("Optimal_Parameters.txt", "a") as file:
            file.write(f"Fitness: {best_result}, Pop_size: {populationSize}, Generations: {numberOfGenerations}\n"
                       f"MP: {best_param[0]}, CP: {best_param[1]} TP: {best_param[2]},"
                       f"CR: {best_param[3]}, CRP: {best_param[4]}\n\n")
