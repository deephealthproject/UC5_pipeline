#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import fire
import gc
from genericpath import exists
import humanize as H
import numpy as np
import pandas as pd
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU, DEV_GPU
from posixpath import join
from tqdm import tqdm
import time
from numpy import count_nonzero as nnz
import os
import pickle
import yaml

import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

from models.recurrent_models import nonrecurrent_model, generate_text_predict_next, generate_text_predict_next_gru
from models.recurrent_models import generate_text
from training.augmentations import *
from utils.vocabulary import Vocabulary
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_eddl_cs(eddl_cs="cpu", eddl_cs_mem="mid_mem", gpu_id=None):
    return  eddl.CS_CPU() if eddl_cs=="cpu" else eddl.CS_GPU(g=gpu_id, mem=eddl_cs_mem)


class TextGenerator:
    def __init__(self, cnn, rnn, dataset, text_dataset, vocab, n_tokens=12, rec_cell_type="lstm", stages=["train", "valid", "test"]):
        self.cnn = cnn
        self.rnn = rnn
        self.ds = dataset
        self.text_ds = text_dataset
        self.vocab = vocab
        self.n_tokens = n_tokens
        self.rec_cell_type = rec_cell_type
        self.stages = stages
        print("CLASS TEXTGENERATOR. Results computed for stages:", self.stages)
    def generate(self):
        stages = {
            "train": ecvl.SplitType.training,
            "valid": ecvl.SplitType.validation,
            "test": ecvl.SplitType.test
        }

        cnn = self.cnn
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        
        rnn = self.rnn
        ds = self.ds
        text_ds = self.text_ds
        results = {}
        
        for stage, split_type in stages.items():
            if stage not in self.stages:
                print(f"stage {stage} skipped")
                continue
            gen_sents = []
            target_sents = []
            print("text generation, stage:", stage)
            ds.SetSplit(split_type)
            ds.ResetBatch()
            ds.Start()
            n_batches = ds.GetNumBatches()
            for bi in range(n_batches):
                if (bi + 1) % 100 == 0:
                    print(f"batch {bi+1} / {n_batches}")

                I, X, Y = ds.GetBatch()
                image_ids = [sample.location_[0] for sample in I]
                texts = text_ds.loc[image_ids, "target_text"].tolist()
                # texts = np.array(texts.tolist()).astype(np.float32)
                cnn.forward([X])
                cnn_semantic = eddl.getOutput(cnn_out)
                cnn_visual = eddl.getOutput(cnn_top)
                
                if self.rec_cell_type == "lstm":
                    gen_sentence = generate_text_predict_next(rnn, texts, self.n_tokens, visual_batch=cnn_visual, semantic_batch=cnn_semantic, dev=False)
                elif self.rec_cell_type == "gru":
                    gen_sentence = generate_text_predict_next_gru(rnn, texts, self.n_tokens, visual_batch=cnn_visual, semantic_batch=cnn_semantic, dev=False)
                else:
                    assert False, f"unknown recurrent cell type: {self.rec_cell_type}"
                # gen_s = np.array(gen_sentence)
                gen_sents.append(gen_sentence)
                target_sents.append(texts)
            #< for over batches
            ds.Stop()
            results[stage] = (np.concatenate(gen_sents, axis=0), np.concatenate(target_sents, axis=0))                
        #< for over stage
        
        dfs = [] # stage dfs
        def clean(s):
            r = []
            for w in s:
                r.append(w)
                if w == Vocabulary.EOS:
                    break
            return r

        for stage, (gen_sents, target_sents) in results.items():
            
            #print(gen_sents.shape)
            #print(target_sents.shape)
            clean_generated = []
            clean_targets = []
            for i in range(gen_sents.shape[0]):
                ww = np.squeeze(gen_sents[i,:])
                tt = np.squeeze(target_sents[i,:])
                clean_generated.append(clean(ww))                
                clean_targets.append(clean(tt))
            df = pd.DataFrame({"generated_i": clean_generated, "target_i": clean_targets})
            df["stage"] = stage
            dfs.append(df)
            print(df)
        results = pd.concat(dfs, axis=0)
        
        def decode(tokens):
            return " ".join([self.vocab.idx2word[t] for t in tokens])
        
        results["generated"] = results.generated_i.apply(decode)
        results["target"] = results.target_i.apply(decode)
        print(results.sample(5))

        smoothing_function = SmoothingFunction()
        smooth = smoothing_function.method3
        # sfn  = [smoothing_function.method2, smoothing_function.method3, smoothing_function.method4, smoothing_function.method5]
        # names = ["_m2", "", "_m3", "_m4", "_m5"]

        results["bleu_1"] = results[["target", "generated"]].apply(lambda x: 
                sentence_bleu(
                        [x[0].split(" ")], 
                        x[1].split(" "), 
                        weights=(1, 0, 0, 0), smoothing_function=smooth), axis=1)

        def bleu2(target, generated):
            # print("target:", target)
            # print("generated:", generated)
            # print(type(target))
            # print(type(generated))
            target = target.split(" ")
            generated = generated.split(" ")
            # print(type(target))
            # print(type(generated))
            score = sentence_bleu([target], generated, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            # print(score)
            return score

            
        results["bleu_2"] = results[["target", "generated"]].apply(lambda x: 
                    bleu2(x[0], x[1]), axis=1)

        results["bleu_3"] = results[["target", "generated"]].apply(lambda x: 
            sentence_bleu(
                    [x[0].split(" ")], 
                    x[1].split(" "), 
                    weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth), axis=1)

        results["bleu_4"] = results[["target", "generated"]].apply(lambda x: 
            sentence_bleu(
                    [x[0].split(" ")], 
                    x[1].split(" "), 
                    weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth), axis=1)
        return results
        
#<


def load_ecvl_dataset(filename, config):
    _, test_augs = get_augmentations(config["dataset_name"], config["img_size"])
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

def generate_for_run(run_fld, dataset, text_dataset, vocab, config):
    n_words = len(vocab.idx2word)
    cnn_fn = join(run_fld, "best_cnn.onnx")

    cnn = eddl.import_net_from_onnx_file(cnn_fn)
    eddl.build(cnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs(config["eddl_cs"], config["eddl_cs_mem"], config["gpu_id"]) , init_weights=False)  # losses, metrics
    eddl.set_mode(cnn, 0)

    visual_dim = eddl.getLayer(cnn, "top").output.shape[1]
    semantic_dim = eddl.getLayer(cnn, "cnn_out").output.shape[1]

    print("visual_dim = ", visual_dim)
    print("semantic_dim = ", semantic_dim)

    n_words = len(vocab.idx2word)

    rnn_fn = join(run_fld, "best_rnn.onnx") if not config["dev"] else join(run_fld, "dev_checkp.onnx")

    trained_rnn = eddl.import_net_from_onnx_file(rnn_fn)
    print("DESERIALIZED RNN")
    eddl.summary(trained_rnn)

    eddl.build(trained_rnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs() , init_weights=False)  # losses, metrics

    rnn = nonrecurrent_model(visual_dim=visual_dim, semantic_dim=semantic_dim, vs=n_words, emb_size=512, lstm_size=512, rec_cell_type=config["rec_cell_type"])
    eddl.build(rnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs() , init_weights=False)  # losses, metrics

    print(f"NON-RECURRENT {config['rec_cell_type']} FOR GENERATION")
    eddl.summary(rnn)

    if config["rec_cell_type"] == "lstm":
        layers_to_copy = [
                    "word_embs",
                    "lstm_cell", "out_dense"
                ]
    elif config["rec_cell_type"] == "gru":
        layers_to_copy = [
                    "word_embs",
                    "gru_cell", "out_dense"
                ]
    else:
        assert False, "wrong recurrent cell type (layer)"

    for l in layers_to_copy:
        eddl.copyParam(eddl.getLayer(trained_rnn, l), eddl.getLayer(rnn, l))

    generator = TextGenerator(cnn, rnn, dataset, text_dataset, vocab, n_tokens=config["n_tokens"], rec_cell_type=config["rec_cell_type"], stages=config["stages"])        
    return generator.generate()
#<

def main(exp_fld, 
        eddl_cs="cpu",
        gpu_id=[1,0,0,0],
        eddl_cs_mem="mid_mem",
        dataset_name="chest-iu",
        img_size=224,
        seed=1,
        shuffle_seed=2,
        bs=128,
        n_tokens=12,
        rec_cell_type="lstm",
        stages=["train", "valid", "test"],
        dev=False):
    print("EXP:", exp_fld)
    config = locals()
    print("TEST RNN, EXP FOLDER:", exp_fld)
    print("REC CELL TYPE: ", rec_cell_type)
    print("RESULTS FOR STAGES:", stages)
    
    with open(join(exp_fld, "idx2label.yaml"), "r") as fin:
        i2l = yaml.load(fin, Loader=yaml.SafeLoader)
    
    print("idx2label, len", len(i2l))
    print("IMAGE LABELS:")
    for i, l in i2l.items():
        print(f"-{ i} -> {l}")
    text_dataset = pd.read_pickle(join(exp_fld, "img_text_dataset.pkl")).set_index("image_filename")
    print("read text dataset, shape", text_dataset.shape)
    with open(join(exp_fld, "vocab.pkl"), "rb") as fin:
        vocab = pickle.load(fin)
    print("read vocabulary, size:", len(vocab.idx2word))

    run_flds = sorted([fn for fn in os.listdir(exp_fld) if fn.startswith("run_")])
    print(f"found {len(run_flds)} runs:")
    for fld in run_flds:
        print(f"\t{fld}")
    print()
    all_results = []
    for fld in run_flds:
        print(f"visiting folder: {fld}")
        dataset = load_ecvl_dataset(join(exp_fld, fld, "ecvl_ds.yml"), config)
        print("ECVL dataset loaded, number of classes:", len(dataset.classes_))

        params_flds = sorted([fn for fn in os.listdir(join(exp_fld, fld)) if fn.startswith("params_")])
        for param_fld in params_flds:
            print(f"\tvisiting inner folder: {param_fld}")
            inner_folder = join(exp_fld, fld, param_fld)
            if not os.path.exists(join(inner_folder, "best_rnn.onnx")):
                print(f"{param_fld} does not contain a trained RNN, skipping")
                continue
            
            results = generate_for_run(inner_folder, dataset, text_dataset, vocab, config)
            results["run"] = inner_folder
            results.to_pickle( join(inner_folder, "rnn_results.pkl") )
            print("FOLDER:", param_fld)
            print(results[["bleu_1", "bleu_2", "bleu_3", "bleu_4", "stage"]].groupby("stage").mean())
            all_results.append(results)
        # for over param  folders
    # for over run folders 
    if len(all_results) > 0:
        results = pd.concat(all_results, axis=0)
        results.to_pickle(join(exp_fld, "rnn_results.pkl"))
        print("results saved:", join(exp_fld, "rnn_results.pkl"))
    else:
        print("no results")
    print("all done.")


if __name__ == "__main__":
    fire.Fire(main)