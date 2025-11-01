import time

import numpy as np


# Function to calculate channel probabilities (Equation 1)
def calculate_channel_probabilities(population):
    costs = population
    probabilities = 1 / (costs + 1e-6)  # Add small value to avoid division by zero
    return probabilities / np.sum(probabilities)


# Roulette Wheel Selection (based on channel probabilities)
def roulette_wheel_selection(probabilities):
    return np.random.choice(len(probabilities))


#  Geyser inspired Algorithm (GEA)
def GEA(population, fobj, VRmin, VRmax, max_iterations):
    n_pop, n_var = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]  # n_pop=60, n_var=30, max_iterations=100, var_min=-100, var_max=100):

    best_fitness = float('inf')
    best_solution = np.zeros((n_pop, 1))
    Convergence_curve = np.zeros((max_iterations, 1))

    t = 0
    ct = time.time()
    for iteration in range(max_iterations):
        # Calculate channel probabilities
        probabilities = calculate_channel_probabilities(population)

        for i in range(n_pop):
            # Select target channel using roulette wheel mechanism
            target_channel = roulette_wheel_selection(probabilities)

            # Determine neighbor based on shortest distance (Eqs. (3)-(4))
            distances = [np.linalg.norm(population[i] - population[j])
                         for j in range(n_pop) if j != i]
            neighbor_index = np.argmin(distances)

            # Update position of Xi using Eq. (4)
            new_position_1 = population[neighbor_index] + np.random.rand(n_var) * (
                    population[target_channel]- population[i])
            new_position_1 = np.clip(new_position_1, lb, ub)
            new_cost_1 = fobj(population[:3])

            # Calculate pressure value (Eq. (6))
            pressure_value = np.random.rand()  # Replace with actual pressure calculation

            # Update channel probability using Eq. (7)
            probabilities = calculate_channel_probabilities(population)

            # Update position of Xi using Eq. (8)
            new_position_2 = population[i] + pressure_value * (
                    population[neighbor_index] - population[i])
            new_position_2 = np.clip(new_position_2, lb, ub)
            new_cost_2 = fobj(new_position_2)

            # Select better solution
            if new_cost_2 < population[i][0]:
                population[i], population[i] = new_position_2, new_cost_2
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iterations - 1][0]
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct
