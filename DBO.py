import time

import numpy as np


def Bounds(x, lb, ub):
    return np.clip(x, lb, ub)


def DBO(pop, fobj, VRmin, VRmax, Max_iter):
    pNum, dim = pop.shape[0], pop.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    x = pop
    fit = fobj(pop[:])
    pFit = fit.copy()
    pX = x.copy()
    XX = pX.copy()

    fMin = np.min(fit)
    bestI = np.argmin(fit)
    bestX = x[bestI, :]

    # Convergence curve
    Convergence_curve = np.zeros(Max_iter)

    ct = time.time()
    for t in range(Max_iter):
        fmax = np.max(fit)
        B = np.argmax(fit)
        worse = x[B, :]
        r2 = np.random.rand()

        # Update for producers
        for i in range(pNum):
            if r2 < 0.9:
                r1 = np.random.rand()
                a = 1 if np.random.rand() > 0.1 else -1
                x[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * XX[i, :]  # Equation (1)
            else:
                aaa = np.random.randint(0, 181)
                theta = aaa * np.pi / 180
                if aaa in [0, 90, 180]:
                    x[i, :] = pX[i, :]
                else:
                    x[i, :] = pX[i, :] + np.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # Equation (2)

            x[i, :] = Bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        # Update best position and fitness
        fMMin = np.min(fit)
        bestII = np.argmin(fit)
        bestXX = x[bestII, :]
        R = 1 - t / Max_iter

        # Generate Xnew based on bestXX and bestX
        Xnew1 = Bounds(bestXX * (1 - R), lb, ub)
        Xnew2 = Bounds(bestXX * (1 + R), lb, ub)
        Xnew11 = Bounds(bestX * (1 - R), lb, ub)
        Xnew22 = Bounds(bestX * (1 + R), lb, ub)

        # Update other individuals based on specific equations
        for i in range(pNum):
            x[i, :] = bestXX + (np.random.rand(dim) * (pX[i, :] - Xnew1) + np.random.rand(dim) * (pX[i, :] - Xnew2))
            x[i, :] = Bounds(x[i, :], Xnew1, Xnew2)
            fit[i] = fobj(x[i, :])

        for i in range(10):
            x[i, :] = pX[i, :] + np.random.randn() * (pX[i, :] - Xnew11) + np.random.rand(dim) * (pX[i, :] - Xnew22)
            x[i, :] = Bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        for j in range(pop.shape[0]):
            x[j, :] = bestX + np.random.randn(dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
            x[j, :] = Bounds(x[j, :], lb, ub)
            fit[j] = fobj(x[j, :])

        # Update personal and global best
        XX = pX.copy()
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
