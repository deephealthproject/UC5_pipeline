#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

from models.rnn_module_att import EddlRecurrentModuleWithAttn
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


def load_ecvl_dataset(filename, config):
    train_augs, test_augs = get_augmentations(config["dataset_name"], config["img_size"])
    augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
    ecvl.AugmentationParam.SetSeed(config["seed"])
    ecvl.DLDataset.SetSplitSeed(config["shuffle_seed"])
    print("loading dataset:", filename)
    
    if config["eddl_cs"] == "cpu":
        num_workers = 16
    else:
        num_workers = 8 if nnz(config["gpu_id"]) == 1 else 8 * nnz(config["gpu_id"])
    print(f"using num workers = {num_workers}")
    
    dataset = ecvl.DLDataset(filename, 
                        batch_size=config["bs"], 
                        augs=augs, 
                        ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
                        num_workers=num_workers, queue_ratio_size= 4 * nnz(config["gpu_id"]), 
                        drop_last=drop_last_rnn_gpu) # drop_last defined in training.augmentations
                        
    return dataset
    #<


def do_training(exp_fld, fld, dataset, text_ds, config):
    rec_module = EddlRecurrentModuleWithAttn(dataset, text_ds, exp_fld, fld, config)
    print("recurrent module created")
    results = rec_module.train()

def main(exp_fld, 
        seed=1,
        shuffle_seed=2,
        n_epochs=1,
        bs=32,
        description="na",
        eddl_cs="cpu",
        gpu_id=[1,0,0,0],
        eddl_cs_mem="mid_mem",
        optimizer="adam", 
        lr=1e-04, 
        early_break=False,
        cnn_pretrained=["resnet18"],
        dataset_name="chest-iu",
        img_size=224,
        verbose=False,
        load_file=None, # file with the trained RNN
        cnn_file=None, # file with the trained CNN
        emb_size=512,
        lstm_size=512,
        n_rnn_cells=1,
        remote_log=True,
        semantic_thresholds=False,
        prefix = None,  # prefix for saved models
        eb_from_epoch = 1000, # epoch from which to start early breaking
        is_timing=False,
        one_param_only=False,
        dev=True):
    config = locals()
    print("RECURRENT MODULE, EXP FOLDER:", exp_fld)
    text_ds_fn = join(exp_fld, "img_text_dataset.pkl")
    text_ds = pd.read_pickle(text_ds_fn).set_index("image_filename")
    print("TEXT DATASET")
    print(text_ds.head())
    print(f"read dataset with text encoding from {text_ds_fn}\n\t shape: {text_ds.shape}")
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
            do_training(exp_fld, join(exp_fld, fld, param_fld), dataset, text_ds, config)
            
            if one_param_only:
                print("breaking, only visiting one param folder")
                break # only 1 params
        # for over param  folders
        # print("breaking, only visiting one run folder")
        # break
    # for over run folders          





if __name__ == "__main__":
    fire.Fire(main)