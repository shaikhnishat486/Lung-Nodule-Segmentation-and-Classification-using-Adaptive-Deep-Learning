import time

import numpy as np


def update_position_foraging(x, sa, r, i, dim):
    # Calculate new position in the foraging phase (Eq. 5)
    I = np.random.uniform(-1, 1, size=dim)
    new_x = x + r * (sa - I * x)
    return new_x


def update_position_digging(x, lb, ub, r, t, T, dim):
    # Calculate new position in the digging phase (Eq. 7)
    new_x = x + (1 - 2 * r) * ((ub - lb) / (t + 1))
    return new_x


def AOA(x, obj_func, VRmin, VRmax, max_iter):
    pop_size, dim = x.shape[0], x.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    fitness = obj_func(x[:])
    best_idx = np.argmin(fitness)
    best_solution = x[best_idx, :]
    best_fitness = float('inf')

    Convergence_curve = np.zeros((max_iter, 1))

    ct = time.time()
    for t in range(1, max_iter + 1):
        r = np.random.rand()  # Random coefficient for position update
        for i in range(pop_size):
            # Foraging Phase (Exploration)
            candidates = [k for k in range(pop_size) if k != i and fitness[k] < fitness[i]]
            if candidates:
                target_idx = np.random.choice(candidates)
                sa = x[target_idx, :]
                new_x = update_position_foraging(x[i, :], sa, r, i, dim)
                new_x = np.clip(new_x, lb, ub)  # Ensure bounds
                new_fitness = obj_func(new_x)

                # Update position if new fitness is better
                if new_fitness < fitness[i]:
                    x[i, :] = new_x
                    fitness[i] = new_fitness

            # Digging Phase (Exploitation)
            new_x = update_position_digging(x[i, :], lb, ub, r, t, max_iter, dim)
            new_x = np.clip(new_x, lb, ub)
            new_fitness = obj_func(new_x)

            # Update position if new fitness is better
            if new_fitness < fitness[i]:
                x[i, :] = new_x
                fitness[i] = new_fitness

        # Update the best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = x[best_idx, :]
            best_fitness = fitness[best_idx]
        Convergence_curve[0] = best_fitness
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
