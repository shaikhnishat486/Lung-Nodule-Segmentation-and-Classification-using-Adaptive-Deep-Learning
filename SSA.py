import time
import numpy as np


# Sparrow Search Algorithm (SSA)
def SSA(pop, fobj, VRmin, VRmax, Max_iter):
    pNum, dim = pop.shape[0], pop.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    x = pop
    fit = np.array([fobj(ind) for ind in x])
    pFit = fobj(pop[:])
    pX = x.copy()
    fMin = float('inf')
    bestX = np.zeros((dim, 1))

    # Start updating the solutions
    Convergence_curve = np.zeros(Max_iter)
    t = 0
    ct = time.time()
    for t in range(Max_iter):
        # Sorting
        sortIndex = np.argsort(pFit)
        fmax = np.max(pFit)
        B = np.argmax(pFit)
        worse = x[B, :]

        r2 = np.random.rand()

        # Update producer positions
        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand()
                x[sortIndex[i], :] = pX[sortIndex[i], :] * np.exp(-i / (r1 * Max_iter))
                x[sortIndex[i], :] = np.clip(x[sortIndex[i], :], lb, ub)
                fit[sortIndex[i]] = fobj(x[sortIndex[i], :])
        else:
            for i in range(pNum):
                x[sortIndex[i], :] = pX[sortIndex[i], :] + np.random.randn() * np.ones(dim)
                x[sortIndex[i], :] = np.clip(x[sortIndex[i], :], lb, ub)
                fit[sortIndex[i]] = fobj(x[sortIndex[i], :])

        # Find best in the current population
        fMMin = np.min(fit)
        bestII = np.argmin(fit)
        bestXX = x[bestII, :]

        # Update joiner positions
        for i in range(pNum):
            A = (np.floor(np.random.rand(dim) * 2) * 2 - 1).astype(int)
            if i > (pop.shape[0] / 2):
                x[sortIndex[i], :] = np.random.randn() * np.exp((worse - pX[sortIndex[i], :]) / (i ** 2))
            else:
                x[sortIndex[i], :] = bestXX + np.abs(pX[sortIndex[i], :] - bestXX)
            x[sortIndex[i], :] = np.clip(x[sortIndex[i], :], lb, ub)
            fit[sortIndex[i]] = fobj(x[sortIndex[i], :])

        # Update personal bests
        for i in range(pop.shape[0]):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]
            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i, :]

        Convergence_curve[t] = fMin
        ct = time.time() - ct

    return fMin, Convergence_curve, bestX, ct
