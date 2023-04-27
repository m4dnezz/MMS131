import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

############################################
# Made by Niclas Persson 2023/04/26
###########################################

# Change these parameters as desired
filename = "data_ga.txt"
populationSize = 200  # Needs to be even
numberOfGenerations = 1000
tournamentProbability = 0.65
crossoverProbability = 0.2
mutationProbability = 0.3
creepProbability = 0.8
creepRate = 0.03

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


def init_ind(parameters=6, mini=-maximumParameterValue, maxi=maximumParameterValue):
    ind = []
    for i in range(parameters):
        ind.append(random.uniform(mini, maxi))
    return ind


def g_fun(ind, x, y):
    approx = (1 + ind[0] * x + ind[1] * x ** 2 + ind[2] * x ** 3) / (1 + ind[3] * y + ind[4] * y ** 2 + ind[5] * y ** 3)
    return approx


def calc_error(approx, gdata):
    error = np.sqrt(1 / populationSize * np.sum((approx - gdata)**2))
    return error


def calc_fitness(error):
    return np.exp(-error)
#######################


def initialize_population(populationSize=100, numberOfParameters=6, empty=False):
    if not empty:
        pop = np.random.uniform(-maximumParameterValue, maximumParameterValue + 0.00001, size=(populationSize, numberOfParameters))
    else:
        pop = np.zeros(shape=(populationSize, numberOfParameters))
    return pop


def evaluate_individual(chromosome, functionData):
    alldata = functionData
    approx = g_fun(chromosome, alldata[0], alldata[1])
    error = calc_error(approx, alldata[2])
    fitness = calc_fitness(error)
    return fitness


def tournament_select(fitness, tournamentProbability, size=2):
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


def cross(chromosome_1, chromosome_2, p_cross):
    if random.random() > p_cross:
        return chromosome_1, chromosome_2
    else:
        cut = random.randint(0, 6)
        new_ind_1 = np.concatenate([chromosome_1[cut:, ], chromosome_2[0:cut]])
        new_ind_2 = np.concatenate([chromosome_2[cut:, ], chromosome_1[0:cut]])
        return new_ind_1, new_ind_2


def mutate(originalchromosome, mutationprobability, creepprobability, creeprate):
    newchromosome = []
    for gene in originalchromosome:
        # Do not perform any mutation
        if random.random()  > mutationprobability:
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


def main():
    # Add nice progress-bar #REMOVE BEFORE HADNING IN

    # Initialize everything
    xdata, ydata, gdata, alldata = importdata(filename)
    population = initialize_population()
    igeneration = 0
    maximum_fitness = []
    best_chromosome = None
    x = []
    pbar = tqdm(total=numberOfGenerations, position=0, leave=True, colour="green")

    # Run the evolution
    while igeneration < numberOfGenerations:
        # Reset parameters for each population
        fitness_population = []
        best_index = None
        fitness = 0

        pbar.update(1)

        # Evaluate generation
        for i in range(len(population)):
            f_ind = evaluate_individual(population[i], alldata)
            fitness_population.append(f_ind)
            if f_ind > fitness:
                fitness = f_ind
                best_index = i

        maximum_fitness.append(fitness)

        # Save best chromosome
        best_chromosome = population[best_index]

        # Generate temporary population
        temp_pop = initialize_population(populationSize, empty=True)

        # Transfer from old population to temporary
        for m in range(0, populationSize, 2):
            # Select two individuals based tournament selection
            ind_1 = tournament_select(fitness_population, tournamentProbability)
            ind_2 = tournament_select(fitness_population, tournamentProbability)

            # Remove them from population
            np.delete(fitness_population, ind_1)
            np.delete(fitness_population, ind_2)

            # Extract chromosomes from index
            chromosome_1 = population[ind_1]
            chromosome_2 = population[ind_2]

            # Perform crossover
            new_individual_pair = cross(chromosome_1, chromosome_2, crossoverProbability)
            temp_pop[m] = new_individual_pair[0]
            temp_pop[m + 1] = new_individual_pair[1]

        # Mutate temp poulation
        for n in range(1, populationSize):
            original_chromosome = temp_pop[n]
            mutated_chromosome = mutate(original_chromosome, mutationProbability, creepProbability, creepRate)
            temp_pop[n] = mutated_chromosome

        temp_pop[0] = best_chromosome  # Elitism
        population = temp_pop
        x.append(igeneration)
        igeneration += 1

    print(f"Fitness is: {fitness} with parameters{best_chromosome}")

    with open("results.txt", "a") as file:
        file.write(f"Fitness: {fitness}, Pop_size: {populationSize}, Generations: {numberOfGenerations}\n"
                   f"TP: {tournamentProbability}, COP: {crossoverProbability} MP: {mutationProbability},"
                   f"CP: {creepProbability}, CR: {creepRate}\n \n")

    plt.figure()
    plt.title("GA Performance over generation")
    plt.plot(x, maximum_fitness,'r--', label="Maximum fitness")
    plt.xlim([100, numberOfGenerations])
    plt.ylim([0.4, 1])
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
