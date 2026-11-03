import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from itertools import product

from sko.GA import RCGA

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.hv import HV

from aux_fun import eval_fun, get_media_dp


def select_hp_grid(fun, lb, ub, output_file, size_pops_hps=[50, 100, 200], pcs_hps=[0.7, 0.85, 0.95], pms_hps=[0.01, 0.05, 0.1, 0.2], execs=10, max_aval=100000, th=1e-4, seed_base=1000):
    dps, _ = get_media_dp(fun, lb, ub)
    def fun_obj(x):
        res = eval_fun(fun, x)
        res_normalizada = res/dps
        return np.sum(res_normalizada)

    configs = product(size_pops_hps, pcs_hps, pms_hps)

    resultados = []
    for size_pop, pc, pm in configs:
        iters = max_aval // size_pop

        fit, sucessos = [], []
        for i in range(execs):

            np.random.seed(seed_base + 1 + i)

            ga = RCGA(
            func=fun_obj,
            n_dim=len(lb),
            size_pop=size_pop,
            max_iter=iters,
            prob_cros=pc,
            prob_mut=pm,
            lb=lb,
            ub=ub,
            )
            best_x, best_y = ga.run()

            r = np.asarray(eval_fun(fun, best_x))
            fit.append(best_y)
            sucessos.append(np.sum(r <= th))

        hp_res = {
            "iters" : iters,
            "size_pop": size_pop,
            "prob_cros": pc,
            "prob_mut": pm,
            "mediana_cumprimentos": np.median(sucessos),
            "mediana_fit": np.median(fit)
        }

        resultados.append(hp_res)

        print(hp_res)

    df = pd.DataFrame(resultados)
    df.to_csv(output_file)



def get_ref_point(fronteiras):
    todas_frentes = np.vstack(fronteiras)

    pior_ponto = np.max(todas_frentes, axis=0)
    melhor_ponto = np.min(todas_frentes, axis=0)
    amplitude = pior_ponto - melhor_ponto
    margem = np.where(amplitude > 0, 0.1 * amplitude, 0.1 * np.maximum(np.abs(pior_ponto), 1e-12))

    return pior_ponto + margem


def select_hp_grid_moo(fun, lb, ub, output_file, size_pops_hps=[50, 100, 200], pcs_hps=[0.7, 0.85, 0.95], pms_hps=[0.01, 0.05, 0.1, 0.2], execs=10, max_aval=100000, seed_base=1000):
    class Problema(ElementwiseProblem):

        def __init__(self):
            super().__init__(
                n_var=len(lb),      
                n_obj=len(fun),      
                xl=lb,
                xu=ub
            )
        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = eval_fun(fun, x)

    problema = Problema()

    configs = product(size_pops_hps, pcs_hps, pms_hps)

    resultados = []

    for size_pop, pc, pm in configs:
        n_gen = int(max_aval/size_pop)

        for i in range(execs):
            algoritmo = NSGA2(
                pop_size=size_pop,
                sampling=FloatRandomSampling(),
                crossover=SBX(
                    prob=pc,
                    eta=15),
                mutation=PM(
                    prob=1.0,
                    prob_var = pm,
                    eta=20),
                eliminate_duplicates=True)
            
            res = minimize(problema, algoritmo, termination=("n_gen", n_gen), seed=i, verbose=False)

            hp_res = {
                "size_pop": size_pop,
                "prob_cros": pc,
                "prob_mut": pm,
                "seed": seed_base + 1 + i,
                "n_gen": n_gen,
                "X": res.X,
                "F": res.F
            }

            print(f'{size_pop} | {pc} | {pm} | {n_gen} | {i}')

            resultados.append(hp_res)

    dps, _ = get_media_dp(fun, lb, ub)
    fronts = [resultado["F"] for resultado in resultados]
    fronts_norm = [front / dps for front in fronts]

    ref_point = get_ref_point(fronts_norm)
    indicador_hv = HV(ref_point=ref_point)

    for resultado, front_norm in zip(resultados, fronts_norm):
        hv = indicador_hv.do(front_norm)
        resultado["hipervolume"] = hv

    df_res = pd.DataFrame(resultados)
    df_res_agg = df_res[['n_gen', 'size_pop', 'prob_cros', 'prob_mut', 'hipervolume']].groupby(['n_gen', 'size_pop', 'prob_cros', 'prob_mut']).mean()

    df_res_agg.to_csv(output_file)