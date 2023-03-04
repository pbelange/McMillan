import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def plot_invariant(expr,q,p,eval_at = None,**kwargs):
    K = sp.lambdify((q,p),expr)

    

    if eval_at is not None:
        q_list,p_list = eval_at
        q_list = np.array(q_list)
        p_list = np.array(p_list)

        levels = K(q_list,p_list)
        
        kwargs.update({'levels':np.sort(levels)})

        multiplier = 1
        window = ((multiplier *q_list,-multiplier *q_list,multiplier *p_list,-multiplier *p_list))
        Q,P = np.meshgrid(  np.linspace(np.min(window),np.max(window),500),
                            np.linspace(np.min(window),np.max(window),500))

    else:
        Q,P = np.meshgrid(  np.linspace(-10,10,10000),
                            np.linspace(-10,10,10000))
    plt.contour(Q,P,K(Q,P),**kwargs)