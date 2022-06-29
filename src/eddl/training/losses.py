#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#


import numpy as np

def weighted_cel(inputs):
    preds = inputs[0]
    target = inputs[1]

    p = np.array(preds)
    t = np.array(target)
    p = np.reshape(p, (1, -1))
    t = np.reshape(t, (1, -1))
    t_pos = t[t > 0]
    t_neg = t[t == 0]
    p_pos = p[t > 0]
    p_neg = p[t == 0]
    n_pos = len(t_pos)
    n_neg = len(t_neg)
    w_pos = (n_pos + n_neg) / n_pos
    w_neg = (n_pos + n_neg) / n_neg
    loss_p = w_pos * np.sum(-1.0 * np.log(p_pos))
    loss_n = w_neg * np.sum(-1.0 * np.log(1.0 - p_neg))
    loss = loss_p + loss_n
    return loss
