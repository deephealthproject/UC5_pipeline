#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from posixpath import join



def label_imbalance(df):
    lab_n1 = df.sum(axis=0)  # frequency of the labels in the dataset
    lab_n0 = df.shape[0] - lab_n1

    lds = pd.concat([lab_n1, lab_n0], axis=1)
    lds.columns = ["n1", "n0"]
    lds.index = lds.index.rename("label")

    # all of the following based on the fact that the minority class is 1
    lds["IRLbl"]  = lds.n1.max(axis=0) / lds.n1
    lds["ImR"] = lds.n0 / lds.n1  # it is max / min, but here majority class is 0

    mean_irlbl = lds.IRLbl.mean()
    lds["m_IRLbl"] = mean_irlbl
    lds["m_ImR"] = lds.ImR.mean()
    
    # variation
    sigma =  (lds.IRLbl - mean_irlbl)**2  
    sigma = sigma.sum()
    sigma = sigma / (lds.shape[0]-1)
    sigma = np.sqrt(sigma)
    # print("sigma:", sigma.shape)
    cvir = sigma / mean_irlbl
    # print("cvir", cvir)
    lds["CVIR"] = cvir

    return lds
#< label_imbalance

def compute_scumble(ds, dev=False):
    li = label_imbalance(ds)
    enc = ds.to_numpy()  # rows: images, columns: labels (1-hot encoded)

    n_labels = np.sum(enc, axis=1)  # vectrized, number of labels per image lambda i: SUM_{j over labels} y_{ij}
    freqs = np.sum(enc, axis=0) # frequency of the labels in the dataset
    
    irlbl = li.IRLbl.to_numpy()  # per label IRLbl measure
    if dev:
        print("enc:", enc.shape)
        print("irlbl:", irlbl.shape)
        print("n_labels:", n_labels.shape)
        # print(irlbl)

    # first step, 
    # for each image, prod of the irlbl of the labels
    irlbl_1 = irlbl.reshape( (1, -1) )
    product = np.multiply(enc, np.repeat(irlbl_1, enc.shape[0], axis=0))
    # substituing zero with one so we can multiply the values across columns
    product2 = np.where(product != 0, product, 1)
    P = product2.prod(axis=1)
    P = P**(1.0 / enc.shape[1])
    
    m_irlbl_img = product.sum(axis=1) / n_labels # each row in product has 0 or the irlbl of the label, divided by num labels in each image

    P = P / m_irlbl_img
    scumble_ins = 1 - P
    scumble = scumble_ins.mean()

    # variation
    squared = (scumble_ins - scumble)**2 / (ds.shape[0] - 1)
    scumble_sigma = np.sqrt(squared.sum())
    scumble_cv = scumble_sigma/scumble if scumble > 0 else 0

    # scumble_lbl
    scumble_ins_1 = scumble_ins.reshape( (-1, 1))
    num = np.multiply(enc, scumble_ins_1)
    num = num.sum(axis=0)
    scumble_lbl = np.divide(num, freqs)

    return scumble, scumble_ins, scumble_cv, scumble_lbl

