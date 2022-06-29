#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import fire
import humanize as H
import numpy as np
import os
import pandas as pd
from posixpath import join
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from sklearn.model_selection import ParameterGrid
from yaml import dump
from numpy import count_nonzero as nnz
from models.cnn_module import EddlCnnModule_ecvl
from training.augmentations import *
import time
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU, DEV_GPU
from sklearn.metrics import roc_curve, accuracy_score
import yaml

def load_ecvl_dataset(filename, config):
    train_augs, test_augs = get_augmentations(config["dataset_name"], config["img_size"])
    augs = ecvl.DatasetAugmentations(augs=[test_augs, test_augs, test_augs])
    ecvl.AugmentationParam.SetSeed(config["seed"])
    ecvl.DLDataset.SetSplitSeed(config["shuffle_seed"])
    print("loading dataset:", filename)
    
    if config["eddl_cs"] == "cpu":
        num_workers = 8
    else:
        num_workers = 8 if nnz(config["gpu_id"]) == 1 else 4 * nnz(config["gpu_id"])
    print(f"using num workers = {num_workers}")
    print("using batch size:", config["bs"])
    dataset = ecvl.DLDataset(filename, 
                        batch_size=config["bs"], 
                        augs=augs, 
                        ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
                        num_workers=num_workers, queue_ratio_size= 4 * nnz(config["gpu_id"]), 
                        drop_last={"training": False, "validation": False, "test": False}) # drop_last defined in training.augmentations
                        
    return dataset
    #<


def compute_per_label_th(predictions, targets, i2l):
    print("predictions has shape:", predictions.shape)
    print("targets has shape:", targets.shape)

    # predictions and targets have shape n_examples x n_labels
    # we need to transpose them
    predictions = np.transpose(predictions)
    targets = np.transpose(targets)
    
    

    auc_acc = []
    J_acc = []

    auc_ths = []
    J_ths = []
    label_names = []
    baselines = []

    target_size = predictions.shape[1]
    for i in range(predictions.shape[0]):  # for i over labels
        y_est = predictions[i, :]  # these are the predictions made by the cnn
        y = targets[i, :]  # there are the true target values 
        fpr, tpr, thresholds = roc_curve(y, y_est)  # check threshoold
        n_zeros = nnz(y)
        n_ones = target_size - n_zeros
        baseline = np.max([n_zeros, n_ones]) / target_size
        baselines.append(baseline)
        # auc
        crit = np.sqrt(tpr * (1 - fpr) )
        m2 = thresholds[np.argmax(crit)]
        auc_ths.append(m2)
        y_est1 = np.where(y_est > m2, 1, 0)
        auc_acc.append(accuracy_score(y_est1, y) * 100)

        # Youden
        J = tpr - fpr
        ij = np.argmax(J)
        th_j = thresholds[ij]
        J_ths.append(th_j)
        y_est4 = np.where(y_est > th_j, 1, 0)
        J_acc.append(accuracy_score(y_est4, y)*100)
        label_names.append(i2l[i])

        print(f"*** label {i}: {i2l[i]}")
        print(f"  - auc1:", auc_acc[-1])
        print(f"  - Youden:", J_acc[-1])
        print(f"auc {m2}, youden {th_j}")
    
    d = {"label": label_names, "auc_t": auc_ths, "auc_acc": auc_acc, "youden_t": J_ths, "youden_acc": J_acc, "majority label" : baselines}
    df = pd.DataFrame.from_dict(d)
    print(df)
    return df
#<



def do_test(param_fld, dataset, i2l, config):
    print("testing in:", param_fld)
    cnn = eddl.import_net_from_onnx_file( join(param_fld, "best_cnn.onnx"))
    eddl.build(
        cnn,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(mem="full_mem"),  # if args.gpu else eddl.CS_CPU(mem=args.mem),
        False  # do not initialize weights to random values
    )
    # cnn.resize(1)  # 1 -> batch size
    eddl.summary(cnn) 
    eddl.set_mode(cnn, 0)
    layer = eddl.getLayer(cnn, "cnn_out")

    predictions = {"train": [], "valid": [], "test": []}
    targets = {"train": [], "valid": [], "test": []}
    stages = {"train": ecvl.SplitType.training, "valid": ecvl.SplitType.validation, "test": ecvl.SplitType.test}

    for stage, st in stages.items():
        print(f"prediction on: {stage}, {st}")
        dataset.SetSplit(st)
        dataset.Start()
        for bi in range(dataset.GetNumBatches()):
            print(".", end="", flush=True)
            _, X, Y = dataset.GetBatch()
            
            eddl.forward(cnn, [X])
            p = np.array(eddl.getOutput(layer), copy=True)
            t = np.array(Y, copy=True)
            # last batch is not complete, so we exclude empty elements (they have all 0 in the target values)
            if bi == dataset.GetNumBatches() - 1:
                sum_t = np.sum(t, axis=1)
                valid = sum_t > 0
                t = t[valid]
                p = p[valid]
            if bi == 0:
                print("predictions ", p.shape)
                print("targets", t.shape)
            predictions[stage].append(p)
            targets[stage].append(t)
                    
        print()
        dataset.Stop()
    print("done with predictions")
    #<
    trva_preds = predictions["train"] + predictions["valid"]
    te_preds = predictions["test"]
    print("len train/valid predictions:", len(trva_preds))
    print("len test predictions:", len(te_preds))
    trva_preds = np.concatenate(trva_preds, axis=0)
    te_preds = np.concatenate(te_preds, axis=0)
    print("train/valid preds shape:", trva_preds.shape)
    print("test preds shape:", te_preds.shape)
    trva_targets = np.concatenate(targets["train"] + targets["valid"], axis=0)
    te_targets = np.concatenate(targets["test"], axis=0)
    trva_df = compute_per_label_th(trva_preds, trva_targets, i2l)
    te_df = compute_per_label_th(te_preds, te_targets, i2l)

    def save_results(df, fname):
        df.to_csv(join(param_fld, fname + ".csv"), index=False)
        df.to_pickle(join(param_fld, fname + ".pkl"))
    
    print("saving results in", param_fld)
    for data, fn in zip([trva_df, te_df], ["trva_thresholds", "te_thresholds"]):
        save_results(data, fn)




def main(exp_fld, 
        bs=128,
        eddl_cs="cpu",
        gpu_id=[1,0,0,0],
        eddl_cs_mem="full_mem",
        cnn_pretrained=["resnet18"],
        dataset_name="mimic",
        img_size=224,
        seed=1,
        shuffle_seed=2,
        dev=True):
    
    config = locals()
    print("TEST CNN, EXP FOLDER:", exp_fld)
    with open(join(exp_fld, "idx2label.yaml"), "r") as fin:
        i2l = yaml.load(fin, Loader=yaml.SafeLoader)
    print("idx2label, len", len(i2l))
    print("IMAGE LABELS")
    for i, l in i2l.items():
        print(f"-{i} -> {l}")
    
    run_flds = sorted([fn for fn in os.listdir(exp_fld) if fn.startswith("run_")])
    print(f"found {len(run_flds)} training datasets:")
    for fld in run_flds:
        print(f"\t{fld}")
    print()
    for fld in run_flds:
        print(f"visiting folder: {fld}")
        dataset = load_ecvl_dataset(join(exp_fld, fld, "ecvl_ds.yml"), config)
        print("ECVL dataset loaded, number of classes:", len(dataset.classes_))
        params_flds = sorted([fn for fn in os.listdir(join(exp_fld, fld)) if fn.startswith("params_")])
        for param_fld in params_flds:
            print(f"\tvisiting inner folder: {param_fld}")
            # ecvl_ds_fn = join(exp_fld, fld, "ecvl_ds.yml")
            do_test(join(exp_fld, fld, param_fld), dataset, i2l, config)
            
        # for over param  folders
    # for over run folders          
    print("all done.")

if __name__ == "__main__":
    fire.Fire(main)