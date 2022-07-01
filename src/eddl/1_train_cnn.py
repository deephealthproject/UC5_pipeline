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

def load_ecvl_dataset(filename, bs, augmentations, gpu):  
    num_workers = 16 * nnz(gpu)  # 8 if nnz(gpu) == 1 else 4 * nnz(gpu)
    # num_workers = 1 * nnz(gpu)
    print(f"--> using num workers = {num_workers}")
    dataset = ecvl.DLDataset(filename, batch_size=bs, augs=augmentations, 
            ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
            num_workers=num_workers, queue_ratio_size= 4 * np.count_nonzero(gpu), drop_last=drop_last)
    return dataset
    #<
def to_list(value):
    return [value] if type(value) is not list else value

def main(
    exp_fld=".",
    dataset_name="chest-iu",
    out_fn="best_cnn.onnx", 
    load_file=None, 
    finetune=[False],
    n_epochs=1, 
    bs=64, 
    optimizer=["adam"], 
    lr=[1e-04], 
    momentum=[0.9], 
    patience=5, 
    patience_kick_in=15,
    seed=1, 
    shuffle_seed=2,
    gpu_id=[1,0,0,0],
    eddl_cs="gpu",
    eddl_cs_mem="mid_mem", 
    verbose=False, 
    pretrained=["resnet18"],
    description="na",
    early_break=True,
    img_size=224,
    activation="sigmoid",
    remote_log=False,
    is_timing=False,  # set to true when taking timings, disable saving of the onnx model
    dev=False):
    
    if dev:
        remote_log = False

    config = locals()

    
    t0 = time.perf_counter()
    runs = sorted([fn for fn in os.listdir(exp_fld) if fn.startswith("run_")])
    n_runs = len(runs)
    if n_runs == 0:
        assert False, f"no runs found in {exp_fld}"
    
    neptune_mode = "offline" if dev or (not remote_log) else "async"
    print(f"flag 'dev' set to {dev}, neptune mode set to {neptune_mode}")

    param_names = ["optimizer", "lr", "momentum", "pretrained", "finetune"]
    parameters = {}
    for k in param_names:
        parameters[k] = to_list(config[k])
        print(f"grid with {k} = {parameters[k]}")
    grid = ParameterGrid(
        parameters
    )
    for k in param_names:
        del config[k]


    print(f"|parameter grid| = {len(grid)}")
    print(f"total trainings: = {len(grid) * n_runs}")
    
    train_augs, test_augs = get_augmentations(dataset_name, config["img_size"])
    augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
    ecvl.AugmentationParam.SetSeed(seed)
    ecvl.DLDataset.SetSplitSeed(shuffle_seed)
    
    all_results = []
    for run_i, run_fld in enumerate(runs):
        t1 = time.perf_counter()
        print(f"*** bootstrap iteration: {run_i+1} / {n_runs}: start")

        run_fld = join(exp_fld, run_fld)
        print("")
        print(f"run {run_i+1}/{n_runs}")
        for param_i, params in enumerate(grid):
            print(f"{param_i+1}/{len(grid)} start")
            for k, v in params.items():
                print(f"- GRID {param_i} {k} = {v}")
            print()
            out_fld = join(run_fld, f"params_{param_i}")

            run = None
            if remote_log:
                import neptune.new as neptune
                run = neptune.init(project="thistlillo/UC5-DeepHealth", mode = neptune_mode)
                run["description"] = f"param_{param_i+1}/{len(grid)}_split_{run_i+1}/{n_runs}, {description}"
                run["folder"] = out_fld
            
            os.makedirs(out_fld, exist_ok=True)
            config["out_fld"] = out_fld
            params["out_fld"] = out_fld
            with open(join(out_fld, "params.yml"), "w") as fout:
                dump(params, fout, default_flow_style=None)
            with open(join(out_fld, "config.yml"), "w") as fout:
                dump(config, fout, default_flow_style=None)
            
            
            # load dataset
            ds_fn = join(run_fld, "ecvl_ds.yml")
            print("dataset loaded")
            print(f"loading ecvl dataset {ds_fn} from yaml file")
            print(f"Creating dataset with batch size: {config['bs']}, {type(config['bs'])}")
            print(f"GPU ids: {gpu_id}")
            
            dataset = load_ecvl_dataset(ds_fn, config["bs"], augs, gpu=gpu_id)
            print("dataset loaded")
            opt_name = params["optimizer"]
            lr = params["lr"]
            pretrained = params["pretrained"]
            finetune = params["finetune"]
            momentum = params["momentum"]
            cnn_module = EddlCnnModule_ecvl(dataset, opt_name, lr, pretrained, finetune, momentum, config, neptune_run=run, name=f"r{run_i}_p{param_i}")
            results = cnn_module.train()
            results["params"] = param_i
            all_results.append(results)
            print(f"{param_i+1}/{len(params)} completed")
        #<
        print(f"* bootstrap iteration: {run_i+1} / {n_runs}: end")
        print(f"*\t time: {H.precisedelta(time.perf_counter() - t0)}")
        print("")

    all_results = pd.concat(all_results, axis=0)
    print("all results, shape:", all_results.shape)
    all_results.to_csv(join(exp_fld, "all_results.csv"), index=False)    
    all_results.to_pickle(join(exp_fld, "all_results.pkl"))


# -------------------
if __name__ == "__main__":
    fire.Fire(main)
