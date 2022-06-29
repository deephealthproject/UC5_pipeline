#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import json
import numpy as np
import os
import pandas as pd
import pickle
from posixpath import join
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pickle
import yaml

# from utils.data_partitioning import DataPartitioner
#import text.reports as reports
# from text.encoding import SimpleCollator, StandardCollator
# from pt.uc5_dataset import Uc5ImgDataset
# import utils.misc as mu

from lightning.dataset import MultiModalDataset, ImageTransforms
from lightning.text_collation import collate_fn_n_sents
class Uc5DataModule(LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.l1normalization = True
        self.img_fld = conf["img_fld"]
        self.exp_fld = conf["exp_fld"]
        self.img_size = conf["img_size"]
        self.img_transforms = ImageTransforms(dataset="chest-iu")
        with open( join(self.exp_fld, "img_dataset.pkl"), "rb") as fin:
            self.img_ds = pickle.load(fin)
        with open( join(self.exp_fld, "img_text_dataset.pkl"), "rb") as fin:
            self.text_ds = pickle.load(fin)
        
        self.train_dl, self.val_dl, self.test_dl = None, None, None   # cache of the dataloaders, not used - code commented
        self.train_ids, self.valid_ids, self.test_ids = self._load_data_split()
                
        # section: load files
        #   read all the files here to reduce memory footprint
        with open(join(self.exp_fld, "label2idx.yaml"), "r") as fin:
            self.label2index = yaml.safe_load(fin)
        self.n_classes = len(self.label2index)

        with open(join(self.exp_fld, "idx2label.yaml"), "r") as fin:
            self.index2label = yaml.safe_load(fin)
        
    #< init
    
    def _load_data_split(self):
        split = pd.read_pickle(join(self.exp_fld, "split_0.pkl"))
        train_ids = split[split.split == "train"].filename
        valid_ids = split[split.split == "valid"].filename
        test_ids = split[split.split == "test"].filename
        print(f"data split read from disk. |train|={len(train_ids)}, |valid|={len(valid_ids)}, |test|={len(test_ids)}")
        return train_ids, valid_ids, test_ids

    

    def _filter_tsv_for_split(self, ids):  # train, val or test ids
        subdf = self.tsv[self.tsv.filename.isin(ids)]  # .reset_index(drop=True)
        return subdf        
    #< section: uc5 datasets


    # (self, img_dataset: pd.DataFrame, text_dataset: pd.DataFrame, 
    #         img_fld: str, img_transforms=None, n_classes=None, img_size=224,
    #         n_sentences=1, n_tokens=12, collate_fn=None, verbose=False, l1normalization=True):
    #     super().__init__()

    #> section: pt-lightning methods
    def train_dataloader(self):
        if self.conf["verbose"]:
            print("returning train_dataloader")
        
        train_dataset = MultiModalDataset(img_dataset=self.img_ds.loc[self.train_ids], 
                text_dataset=self.text_ds, img_fld=self.img_fld, img_transforms=self.img_transforms.train_transforms, n_sentences=3, n_tokens=12, collate_fn=collate_fn_n_sents)

        print(f"train dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(train_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"])
    #<
        
    def val_dataloader(self):
        if self.conf["verbose"]:
            print("returning val_dataloader")
        
        val_dataset = MultiModalDataset(img_dataset=self.img_ds.loc[self.valid_ids], 
                text_dataset=self.text_ds, img_fld=self.img_fld, img_transforms=self.img_transforms.test_transforms, n_sentences=3, n_tokens=12, collate_fn=collate_fn_n_sents)
        print(f"val dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(val_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"]) # , num_workers=self.conf["loader_threads"]
    #<

    def test_dataloader(self):
        if self.conf["verbose"]:
            print("returning test_dataloader")
                
        test_dataset = MultiModalDataset(img_dataset=self.img_ds.loc[self.test_ids], 
                text_dataset=self.text_ds, img_fld=self.img_fld, img_transforms=self.img_transforms.test_transforms, n_sentences=3, n_tokens=12, collate_fn=collate_fn_n_sents)
        print(f"test dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(test_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"]) # , num_workers=self.conf["loader_threads"]
    #<    
    # section: pt-lightning methods

# --------------------------------------------------
# main USED ONLY FOR TESTING
def main(in_tsv,
         exp_fld,
         img_fld,
         out_fn="uc5model_default.bin",
         only_images=False,
         train_p=0.7,
         valid_p=0.1,
         seed=1,
         shuffle_seed=2,
         term_column="auto_term",
         text_column="text",
         img_size = 224,
         batch_size=32,
         last_batch="random",
         n_epochs=50,
         n_sentences=5,  # ignored by simple
         n_tokens=10,
         eddl_cs_mem="mid_mem",
         eddl_cs="cpu",
         sgl_lr=0.09,
         sgd_momentum=0.9,
         lstm_h_size=512,
         emb_size=512,
         load_data_split=True,
         preload_images = True,
         verbose=False,
         debug=False,
         dev=False):
    config = locals()
    datamod = Uc5DataModule(config)
    train_dl = datamod.train_dataloader()
    val_dl = datamod.val_dataloader()
    test_dl = datamod.test_dataloader()

    print(f"|train_dataloader|= {len(train_dl)}")
    print(f"|val_dataloader|= {len(val_dl)}")
    print(f"|test_dataloader|= {len(test_dl)}")
    