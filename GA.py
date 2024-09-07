import numpy as np
import random

# Define the matrix
matrix = np.array([
    [7, 9, 2, 6, 5],
    [8, 8, 5, 6, 7],
    [2, 8, 9, 6, 8],
    [7, 2, 6, 3, 9],
    [2, 9, 1, 7, 5]
])

# Parameters for the genetic algorithm
population_size = 10  # The size of the population
mutation_rate = 0.1  # The probability of mutation
generations = 50  # The number of generations to run
n = matrix.shape[0]  # The dimension of the matrix (number of rows/columns)


# Function to create an individual (a permutation of [0,1,2,...,n-1])
def create_individual():
    return random.sample(range(n), n)


# Fitness function: calculate the sum of the selected elements
def fitness(individual):
    return sum(matrix[i, individual[i]] for i in range(n))


# Create the initial population
def create_population():
    return [create_individual() for _ in range(population_size)]


# Selection: tournament selection
def selection(population, scores, k=3):
    selected = random.sample(population, k)
    selected_scores = [scores[population.index(ind)] for ind in selected]
    return selected[selected_scores.index(max(selected_scores))]


# Crossover: single point crossover
def crossover(parent1, parent2):
    point = random.randint(1, n - 2)
    child = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
    return child


# Mutation: swap mutation
def mutate(individual):
    if random.random() < mutation_rate:
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# Genetic Algorithm
def genetic_algorithm():
    # Initialize the population
    population = create_population()

    for generation in range(generations):
        # Calculate fitness scores
        scores = [fitness(individual) for individual in population]

        # Create the next generation
        next_population = []
        for _ in range(population_size):
            # Select two parents
            parent1 = selection(population, scores)
            parent2 = selection(population, scores)
            # Perform crossover
            child = crossover(parent1, parent2)
            # Perform mutation
            child = mutate(child)
            next_population.append(child)

        population = next_population

        # Track the best solution
        best_fitness = max(scores)
        best_individual = population[scores.index(best_fitness)]

    # Return the best found solution
    return best_individual, best_fitness


# Run the genetic algorithm
best_individual, best_fitness = genetic_algorithm()

# Output the best solution
print("Best individual (column selections):", best_individual)
print("Maximum sum:", best_fitness)
