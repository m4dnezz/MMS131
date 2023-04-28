import numpy as np
import random
from matplotlib import pyplot as plt

############################################
# Made by Niclas Persson 2023/04/26
###########################################

# Change these parameters as desired
filename = "data_ga.txt"
populationSize = 200
numberOfGenerations = 100
tournamentProbability = 0.8
crossoverProbability = 0.5
mutationProbability = 0.1
creepProbability = 0.7
creepRate = 0.02

# Do not change these
numberOfParameters = 6
maximumParameterValue = 2


#############################


def importdata(file: str):
    alldata = np.genfromtxt(file)
    xdata = alldata[:, 0]
    ydata = alldata[:, 1]
    gdata = alldata[:, 2]
    return xdata, ydata, gdata, alldata


def InitializePopulation(populationSize, numberOfParameters, empty=False):
    if not empty:
        pop = np.random.uniform(-maximumParameterValue, maximumParameterValue + 0.00001,
                                size=(populationSize, numberOfParameters))
    else:
        pop = np.zeros(shape=(populationSize, numberOfParameters))
    return pop


def EvaluateIndividual(ind, functionData):
    alldata = functionData
    x = alldata[:, 0]
    y = alldata[:, 1]
    g = alldata[:, 2]
    approx = (1 + ind[0] * x + ind[1] * x ** 2 + ind[2] * x ** 3) / (1 + ind[3] * y + ind[4] * y ** 2 + ind[5] * y ** 3)
    error = np.sqrt(np.sum((approx - g) ** 2) * (1 / populationSize))
    fitness = np.exp(-error)
    return fitness


def TournamentSelect(fitness, tournamentProbability):
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


def Cross(chromosome1, chromosome2):
    if random.random() > crossoverProbability:
        return chromosome1, chromosome2
    else:
        cut = random.randint(0, 6)
        new_ind_1 = np.concatenate([chromosome1[0:cut], chromosome2[cut:, ]])
        new_ind_2 = np.concatenate([chromosome2[0:cut], chromosome1[cut:, ]])
        return new_ind_1, new_ind_2


def Mutate(originalChromosome, mutationProbability, creepProbability, creepRate):
    newchromosome = []
    for gene in originalChromosome:
        # Do not perform any mutation
        if random.random() > mutationProbability:
            newchromosome.append(gene)
        # Perform Mutation
        else:
            # Perform Full-range mutation
            if random.random() > creepProbability:
                newchromosome.append(np.random.uniform(-2, 2))
            # Perform Creep mutation
            else:
                mutation = random.uniform(-creepRate / 2, creepRate / 2)
                newchromosome.append(gene + mutation)
    return newchromosome


def main():
    # Initialize everything
    xdata, ydata, gdata, alldata = importdata(filename)
    population = InitializePopulation(populationSize, numberOfParameters)
    igeneration = 0
    maximum_fitness = []
    best_chromosome = None
    x = []
    fitness = None
    # population[0,:] = [-2,0,1,0,0.5,-0.5 ] used to test the "true" parameters

    # Run the evolution
    while igeneration < numberOfGenerations:
        # Reset parameters for each population
        fitness_population = []
        best_index = None
        fitness = 0

        # Evaluate generation
        for i in range(len(population)):
            f_ind = EvaluateIndividual(population[i], alldata)
            fitness_population.append(f_ind)
            if f_ind > fitness:
                fitness = f_ind
                best_index = i

        maximum_fitness.append(fitness)

        # Save best chromosome
        best_chromosome = population[best_index]

        # Generate temporary population
        temp_pop = InitializePopulation(populationSize, numberOfParameters, empty=True)

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
            new_individual_pair = Cross(chromosome_1, chromosome_2)
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

    print(f"Fitness is: {fitness} with parameters{best_chromosome}")

    with open("results.txt", "a") as file:
        file.write(f"Fitness: {fitness}, Pop_size: {populationSize}, Generations: {numberOfGenerations}\n"
                   f"TP: {tournamentProbability}, COP: {crossoverProbability} MP: {mutationProbability},"
                   f"CP: {creepProbability}, CR: {creepRate}\n"
                   f"Chromosome: {best_chromosome} \n \n")

    plt.figure()
    plt.title("GA Performance over generation")
    plt.plot(x, maximum_fitness, 'r--', label="Maximum fitness")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
