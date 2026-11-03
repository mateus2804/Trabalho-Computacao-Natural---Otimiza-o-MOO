import numpy as np

def eval_fun(fun_list, x):
    return [abs(f(x)) for f in fun_list]


def get_media_dp(lista_funcoes, lb, ub, num_amostras_escala=10000):
    parametros_aleatorios = np.random.uniform(lb, ub, size=(num_amostras_escala, len(lb)))
    res = [eval_fun(lista_funcoes, param) for param in parametros_aleatorios]

    return np.std(res, axis=0), np.mean(res, axis=0)