import numpy as np
from Evaluation import net_evaluation, evaluation
from Global_Vars import Global_Vars
from Model_MVit import Model_ViT


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_ViT(data, Tar, sol)
            Eval = evaluation(pred, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[16])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_ViT(data, Tar, sol)
        Eval = evaluation(pred, Tar)
        Fitn = 1 / (Eval[4] + Eval[16])
        return Fitn


def objfun_cls(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_ViT(data, Tar, sol)
            Eval = evaluation(pred, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[16])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_ViT(data, Tar, sol)
        Eval = evaluation(pred, Tar)
        Fitn = 1 / (Eval[4] + Eval[16])
        return Fitn
