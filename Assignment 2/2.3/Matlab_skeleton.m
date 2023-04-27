

 For problem 2.3, you must implement the following functions,
 with input parameters *exactly* as specified.

 (1) InitializePopulation(populationSize, numberOfParameters)
 This function should generate the population as a matrix with # rows =
populationSize
 and # columns = numberOfParameters. Each gene (matrix element) should take a
 value in the range [-2,2] (decimal values, not integers. Use the rand
function!)

 (2) EvaluateIndividual(chromosome, functionData)
 This function should generate (as output) the fitness value; see the problem
 formulation for details. Note: It is not always a good idea to pass a data
 set as a parameter, but here the data set is small, so it is OK.

 (3) TournamentSelect(fitness, tournamentProbability)
 This function should carry out tournament selection (with tournament size =
2)
 with the given value of pTour, and return the index of a single selected
individual.

 (4) Cross(chromosome1, chromosome2)
 This function should carry out single-point crossover, with a randomly
selected
 crossover point, and return a pair of new (offspring) individuals.

 (5) Mutate(originalChromosome, mutationProbability, creepProbability, creepRate)
 This function should carry out mutations, by running through all genes in the
 original chromosome, checking whether or not the gene should be mutated. If
yes,
 apply either (with probability 1-pCreep) a full-range mutation or (with
probability pCreep)
 a creep mutation with the specified creep rate. Then return the mutated
chromosome.

 Also: implement a plot, showing the maximumFitness as a function of the number
of generations.
 You must therefore add code (below) for storing the maximumFitness (i.e. a
vector of values,
 one for each generation).



populationSize = 100;  modify as desired (but should be even)
numberOfGenerations = 10000;  modify as desired
maximumParameterValue = 2;  parameters range from -2 to 2 (do not change)
tournamentProbability = 0.75;  modify as desired
crossoverProbability = 0.75;  modify as desired
mutationProbability = 0.125;  modify as desired
creepProbability = 0.5;  modify as desired
creepRate = 0.01;  modify as desired
functionData = LoadFunctionData();
format long
numberOfParameters = 6;
bestIndividual = zeros(numberOfParameters, 1);  Best values of the a_i and b_i
parameters

 Initialize population

population = InitializePopulation(populationSize, numberOfParameters);
fitness = zeros(populationSize,1);
maximumFitness = 0.0;
for iGeneration = 1:numberOfGenerations
maximumFitnessInCurrentGeneration = 0;

 Evaluate individuals

for i = 1:populationSize
chromosome = population(i,:);
fitness(i) = EvaluateIndividual(chromosome,functionData);  Write this
function
if (fitness(i) > maximumFitnessInCurrentGeneration)
maximumFitnessInCurrentGeneration = fitness(i);
iBestIndividual = i;
if (fitness(i) > maximumFitness)
maximumFitness = fitness(i);
bestIndividual = chromosome;
fprintf('d 12.8f\n',iGeneration, maximumFitness);
end
end
end
 Add plot of maximum fitness, either here or at the end of the
 run. In the latter case, at least make sure to store the
 maximum fitness for each generation, so that you can later make the plot.

 Generate new population, unless the last generation has been evaluated

if (iGeneration < numberOfGenerations)
temporaryPopulation = population;
for i = 1:2:populationSize
i1 = TournamentSelect(fitness,tournamentProbability);  Write this
function
i2 = TournamentSelect(fitness,tournamentProbability);  Write this
function
r = rand;
if (r < crossoverProbability)
chromosome1 = population(i1,:);
chromosome2 = population(i2,:);
newIndividualPair = Cross(chromosome1, chromosome2);  Write this
function
temporaryPopulation(i,:) = newIndividualPair(1,:);
temporaryPopulation(i+1,:) = newIndividualPair(2,:);
else
temporaryPopulation(i,:) = population(i1,:);
temporaryPopulation(i+1,:) = population(i2,:);
end
end
temporaryPopulation(1,:) = population(iBestIndividual,:);  Elitism,
prevents maximum fitness from decreasing.
for i = 2:populationSize
originalChromosome = temporaryPopulation(i,:);
mutatedChromosome = Mutate(originalChromosome,
mutationProbability,creepProbability,creepRate);  Write this function
temporaryPopulation(i,:) = mutatedChromosome;
end
population = temporaryPopulation;
end
end

 Here, you should add printout of the
 final results, i.e. bestIndividual,
 the minimum error etc., nicely formatted,
 as well as a plot of the maximumFitness
 as specified above.
