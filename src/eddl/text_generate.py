import gc
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


import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

from models.recurrent_models import recurrent_lstm_model, nonrecurrent_lstm_model, generate_text_predict_next
from models.recurrent_models import generate_text
from training.augmentations import *
from utils.vocabulary import Vocabulary
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

eddl_cs_mem = "mid_mem"
def get_eddl_cs():
    return  eddl.CS_GPU(g=[0,0,0,1], mem=eddl_cs_mem)


cnn_fn = "/mnt/datasets/uc5/EXPS/eddl/text_generation/best_cnn.onnx"
cnn = eddl.import_net_from_onnx_file(cnn_fn)
eddl.build(cnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs() , init_weights=False)  # losses, metrics
eddl.set_mode(cnn, 0)

visual_dim = eddl.getLayer(cnn, "top").output.shape[1]
semantic_dim = eddl.getLayer(cnn, "cnn_out").output.shape[1]

print("visual_dim = ", visual_dim)
print("semantic_dim = ", semantic_dim)

with open("/mnt/datasets/uc5/EXPS/eddl/text_generation/vocab.pkl", "rb") as fin:
    vocab = pickle.load(fin)
n_words = len(vocab.idx2word)

rnn_fn = "/mnt/datasets/uc5/EXPS/eddl/text_generation/best_rnn.onnx"
trained_rnn = eddl.import_net_from_onnx_file(rnn_fn)
eddl.build(trained_rnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs() , init_weights=False)  # losses, metrics

rnn = nonrecurrent_lstm_model(visual_dim=visual_dim, semantic_dim=semantic_dim, vs=n_words, emb_size=512, lstm_size=512)
eddl.build(rnn, eddl.adam(lr=1e-04), ["binary_cross_entropy"], ["binary_accuracy"], get_eddl_cs() , init_weights=False)  # losses, metrics

print("TRAINED")
eddl.summary(trained_rnn)

print("NON-RECURRENT FOR GENERATION")
eddl.summary(rnn)

layers_to_copy = [
            "word_embs",
            "lstm_cell", "out_dense"
        ]
for l in layers_to_copy:
    eddl.copyParam(eddl.getLayer(trained_rnn, l), eddl.getLayer(rnn, l))


def load_ecvl_dataset(filename, bs, augmentations, n_gpus, drop_last):  
    num_workers = 4 * n_gpus # 8 if nnz(gpu) == 1 else 4 * nnz(gpu)
    print(f"--> using num workers = {num_workers}")
    
    dataset = ecvl.DLDataset(filename, batch_size=bs, augs=augmentations, 
            ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
            num_workers=num_workers, queue_ratio_size= 2 * n_gpus, drop_last=drop_last)
    return dataset

dataset_fn = "/mnt/datasets/uc5/EXPS/eddl/text_generation/ecvl_ds.yml"
train_augs, test_augs = get_augmentations("chest-iu")
augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
n_gpu = 1
drop_last =  drop_last_rnn_gpu
bs = 128

ds = load_ecvl_dataset(dataset_fn, bs, augs, 1, drop_last)

text_ds = pd.read_pickle("/mnt/datasets/uc5/EXPS/eddl/text_generation/img_text_dataset.pkl").set_index("image_filename")

class TextGenerator:
    def __init__(self, cnn, rnn, dataset, text_dataset, vocab, n_tokens=12):
        self.cnn = cnn
        self.rnn = rnn
        self.ds = dataset
        self.vocab = vocab
        self.n_tokens = n_tokens
        
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
        results = {}
        
        for stage, split_type in stages.items():
            gen_sents = []
            target_sents = []
            print("stage:", stage)
            ds.SetSplit(split_type)
            ds.ResetBatch()
            ds.Start()
            for bi in range(ds.GetNumBatches()):
                I, X, Y = ds.GetBatch()
                image_ids = [sample.location_[0] for sample in I]
                texts = text_ds.loc[image_ids, "target_text"].tolist()
                # texts = np.array(texts.tolist()).astype(np.float32)
                cnn.forward([X])
                cnn_semantic = eddl.getOutput(cnn_out)
                cnn_visual = eddl.getOutput(cnn_top)
                gen_sentence = generate_text_predict_next(rnn, texts, self.n_tokens, visual_batch=cnn_visual, semantic_batch=cnn_semantic, dev=False)
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
            print(gen_sents.shape)
            print(target_sents.shape)
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
            return " ".join([vocab.idx2word[t] for t in tokens])
        
        results["generated"] = results.generated_i.apply(decode)
        results["target"] = results.target_i.apply(decode)
        print(results)

        smoothing_function = SmoothingFunction()
        smooth = smoothing_function.method4
        print("using smoothing function:", smooth)

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
        results.to_pickle("rnn_results.pkl")
        print(results[["bleu_1", "bleu_2", "bleu_3", "bleu_4", "stage"]].groupby("stage").mean())
        print("all done")
generator = TextGenerator(cnn, rnn, ds, text_ds, vocab, n_tokens=12)        
generator.generate()
