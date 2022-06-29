#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

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

from models.recurrent_models import recurrent_lstm_model, nonrecurrent_lstm_model
from models.recurrent_models import generate_text
from utils.bleu_eddl import compute_bleu_edll as compute_bleu
from training.early_stopping import UpEarlyStopping, GLEarlyStopping, ProgressEarlyStopping2, PatienceEarlyStopping

# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)


class EddlRecurrentModule_V2:
    def __init__(self, dataset, text_dataset, exp_fld, fld, config):
        self.conf = Bunch(**config)
        print("RNN, prefix for saved models:", self.conf.prefix)
        self.prefix = self.conf.prefix or ""
        
        self.verbose = self.conf.verbose
        # load vocabulary
        with open(join(exp_fld, "vocab.pkl"), "rb") as fin:
            vocab = pickle.load(fin)
        self.voc_size = len(vocab.word2idx)
        
        self.ds = dataset
        self.text_ds = text_dataset
        self.exp_fld = exp_fld
        self.fld = fld
        self.cnn = self.load_cnn()
        
        if self.conf.load_file:
            print(f"loading model from file {self.conf.load_file}")
            self.rnn = self.load_model()
        else:
            self.rnn = self.build_model()
        
        self.rnn2 = None  # non-recurrent version of self.rnn
        
        self.run = None
        if self.conf.remote_log:
            print("remote log set, importing neptune")
            self.run = self.init_neptune()

        self.best_validation_loss = np.inf
        self.best_validation_acc = -np.inf
        # ---
        self.stages = {
            "train": ecvl.SplitType.training,
            "valid": ecvl.SplitType.validation,
            "test": ecvl.SplitType.test
        }
        self.epoch_timing = []
        self.stage_losses = { stage: [] for stage in self.stages }  # "train": [] , "valid": [], "test": []}
        self.stage_accs = { stage: [] for stage in self.stages }
        self.best_loss = {stage: np.inf for stage in self.stages}
        self.best_acc = {stage: -np.inf for stage in self.stages}
        self.timings = []
        upEs = UpEarlyStopping()
        glEs = GLEarlyStopping()
        prEs = ProgressEarlyStopping2()
        patEs = PatienceEarlyStopping(patience=5, is_loss=True)
        self.early_stop_criteria = {"up": upEs, "gl":glEs, "progress":prEs, "patience":patEs}
        self.early_stop_logs = { name: list() for name in self.early_stop_criteria.keys() }

        if self.conf.semantic_thresholds:
            self.semantic_thresholds = pd.read_pickle(join(self.fld, "trva_thresholds.pkl"))["auc_t"].to_numpy()
            print("thresholds loaded, shape:", self.semantic_thresholds.shape)
            # print(self.semantic_thresholds)
        self.dev = self.conf.dev
    #<

    def apply_thresholds(self, values):
        assert values is not None, "values is none"
        assert self.semantic_thresholds is not None, "thresholds is none"
        return np.where(values > self.semantic_thresholds, 1, 0)

    def init_neptune(self):
        import neptune.new as neptune
        if self.conf["dev"]:
            neptune_mode = "debug"
        elif self.conf["remote_log"]:
            neptune_mode = "async"
        else:
            neptune_mode = "offline"
        run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
        run["description"] = "rnn_module"
        return run 
    #<

    def comp_serv(self, eddl_cs=None, eddl_mem=None):
        eddl_cs = eddl_cs or self.conf.eddl_cs
        eddl_mem = eddl_mem or self.conf.eddl_cs_mem

        print("creating computing service:")
        print(f"computing service: {eddl_cs}")
        print(f"memory: {eddl_mem}")
        lsb = 1
        return eddl.CS_GPU(g=self.conf.gpu_id, mem=self.conf.eddl_cs_mem, lsb=lsb) if eddl_cs == 'gpu' else eddl.CS_CPU(th=16, mem=eddl_mem)
    #< 

    def load_cnn(self):
        filename = self.conf.cnn_file or join(self.fld, f"{self.prefix}best_cnn.onnx")
        cnn = eddl.import_net_from_onnx_file(filename)
        print(f"trained cnn read from: {filename}")
        print(f"cnn input shape {cnn.layers[0].input.shape}")
        print(f"cnn output shape {cnn.layers[-1].output.shape}")
        
        eddl.build(cnn, eddl.adam(0.0001), ["softmax_cross_entropy"], ["accuracy"], # not relevant: it is used only in forwarding
            self.comp_serv(), init_weights=False)
        print("cnn model built successfully")
        for name in [_.name for _ in cnn.layers]:
            eddl.setTrainable(cnn, name, False)
        eddl.summary(cnn)
        return cnn
    #<        
        
    def load_model(self, filename=None, for_predictions=False):
        filename = filename or self.conf.load_file
        print(f"loading file from file: {self.conf.load_file}")
        onnx = eddl.import_net_from_onnx_file(self.conf.load_file) 
        return self.build_model(onnx)
    #<


    def create_model_v1(self, visual_dim, semantic_dim, for_predictions=False):
        # cnn
        vs = self.voc_size  # vocabulary size
        if not for_predictions:
            model = recurrent_lstm_model(visual_dim, semantic_dim, vs, self.conf.emb_size, self.conf.lstm_size)
        else:
            assert self.rnn is not None
            model = nonrecurrent_lstm_model(visual_dim, semantic_dim, vs, self.conf.emb_size, self.conf.lstm_size)
        #<
        print(f"recurrent model created, for predictions? {for_predictions}")
        eddl.summary(model)
        return model

    def create_model_v2(self, visual_dim, semantic_dim, for_predictions=False, n_lstm_cells=1, init_v=0.05):
        assert type(n_lstm_cells) is int and n_lstm_cells >= 1
        vs = self.voc_size
        cnn_top_in = eddl.Input([visual_dim], name="in_visual_features")
        cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
        cnn_app = eddl.Concat([cnn_top_in, cnn_out_in], name="co_attention") 
        
        word_in = eddl.Input([vs])
        to_lstm = eddl.ReduceArgMax(word_in, [0])
        to_lstm = eddl.RandomUniform(eddl.Embedding(to_lstm, vs, 1, self.conf.emb_size, mask_zeros=True, name="word_embs"), -init_v, init_v)
        to_lstm = eddl.Concat([to_lstm, cnn_app])
        
        lstms = []
        lstm = eddl.LSTM(to_lstm, self.conf.lstm_size, mask_zeros=True, bidirectional=False, name="lstm_cell0")
        lstms.add(lstm)
        for i in range(1, n_lstm_cells):
            lstm = eddl.LSTM(lstm, self.conf.lstm_size, mask_zeros=True, bidirectional=False, name=f"lstm_cell{i}")
            lstms.add(lstm)
        self.lstms = lstms

        # eddl.setDecoder(word_in)
        out_lstm = eddl.Softmax(eddl.Dense(lstms[-1], vs, name="out_dense"), name="rnn_out")
        print(f"layer lstm, output shape: {out_lstm.output.shape}")
        # model
        rnn = eddl.Model([cnn_top_in, cnn_out_in, word_in], [out_lstm])


    def create_model(self, visual_dim, semantic_dim, for_predictions=False, version=1):
        if version == 1:
            return self.create_model_v1(visual_dim, semantic_dim, for_predictions)
        elif version == 2:
            return self.create_model_v2(visual_dim, semantic_dim, for_predictions)
        else:
            assert False, f"unknown model version: {version}"
    #<

    def get_optimizer(self):
        self.opt_name = self.conf.optimizer
        opt_name= self.opt_name
        if opt_name == "adam":
            print(f"using learning rate: {self.conf.lr}")
            return eddl.adam(lr=self.conf.lr)
        elif opt_name == "cyclic":
            return eddl.sgd(lr=0.001)
        else:
            assert False
    #<

    def build_model(self, mdl=None, for_predictions=False):
        do_init_weights = (mdl is None)  # init weights only if mdl has not been read from file
        if mdl is None:
            assert self.cnn is not None
            visual_dim = eddl.getLayer(self.cnn, "top").output.shape[1]
            semantic_dim = eddl.getLayer(self.cnn, "cnn_out").output.shape[1]            
            rnn = self.create_model(visual_dim, semantic_dim, for_predictions=for_predictions)
        
        optimizer = self.get_optimizer()
        print(f"building recurrent model, initialization? {do_init_weights}")
        eddl.build(rnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], 
            eddl.CS_CPU() if for_predictions else self.comp_serv(), init_weights=do_init_weights)
        
        #for name in [_.name for _ in rnn.layers]:
        #    eddl.initializeLayer(rnn, name)
        
        eddl.summary(rnn)
        print('rnn built - seems ok')
        return rnn
    #<

    def train(self):
        print("TRAINING")
        conf = self.conf
        cnn = self.cnn
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        print(f"cnn, top layer {cnn_top.output.shape}, output layer {cnn_out.output.shape}")
        # rnn = self.rnn

        # ds = self.ds        
        # ds.set_stage("train")
        ds = self.ds
        text_ds = self.text_ds
        print("- training batches:", ds.GetNumBatches(ecvl.SplitType.training))
        print("- validation batches:", ds.GetNumBatches(ecvl.SplitType.validation))
        print("- test batches:", ds.GetNumBatches(ecvl.SplitType.test))
        print("- dev set?", self.dev)
        print()

        n_epochs = conf.n_epochs
        # batch_ids = range(len(ds))
        eb_kick_in = conf.eb_from_epoch
        print("early breaking criteria applied from epoch:", eb_kick_in)

        # self.run["params/activation"] = self.activation_name
        for ei in range(n_epochs):
            print(f"epoch {ei+1} / {n_epochs}")
            epoch_start = time.perf_counter()
            
            # for every stage, performa training, validation and test
            for stage in self.stages.keys():
                eddl_mode = 1 if stage == "train" else 0
                eddl.reset_loss(self.rnn)
                print(f"> STAGE: {stage}, eddl mode: {eddl_mode}")
                eddl.set_mode(self.rnn, eddl_mode)

                print("> \t split:", self.stages[stage])
                
                if conf.is_timing and stage == "test":
                    print("timing mode, skipping test")
                    self.stage_losses["test"].append(0.1)
                    self.stage_accs["test"].append(0.1)
                    continue

                ds.SetSplit(self.stages[stage])
                ds.ResetBatch(shuffle = (stage == "train") )
                ds.Start()
                epoch_loss = 0
                epoch_acc = 0

                for bi in range(ds.GetNumBatches()):
                    if stage == "train":
                        eddl.reset_loss(self.rnn)
                        eddl.zeroGrads(self.rnn)

                    I, X, Y = ds.GetBatch()
                    image_ids = [sample.location_[0] for sample in I]
                    texts = text_ds.loc[image_ids, "target_text"]
                    texts = np.array(texts.tolist()).astype(np.float32)
                    cnn.forward([X])
                    cnn_semantic = eddl.getOutput(cnn_out)
                    cnn_visual = eddl.getOutput(cnn_top)
                    
                    if self.conf.semantic_thresholds:
                        sem = np.array(cnn_semantic)
                        thresholded = self.apply_thresholds(sem)
                        cnn_semantic = Tensor.fromarray(thresholded)
                        
                    Y = Tensor.fromarray(texts)
                    Y = Tensor.onehot(Y, self.voc_size)


                    #print(Y.shape)
                    #X.info()
                    #Y.info()

                    self.inner_forward(cnn_visual, cnn_semantic, Y)


                    
                    # eddl.train_batch(self.rnn, [cnn_visual, cnn_semantic], [Y])
                    
                    eddl.eval_batch(self.rnn, [cnn_visual, cnn_semantic], [Y])
                    loss = eddl.get_losses(self.rnn)[0]
                    acc = eddl.get_metrics(self.rnn)[0]
                    epoch_loss += loss
                    epoch_acc += acc
                    if self.dev and bi == 5:
                        print("breaking at batch index 5 because dev is set")
                        break
                #< for over batches
                print(f"stage {stage} completed")
                ds.Stop()
                epoch_loss = epoch_loss / ds.GetNumBatches()
                epoch_acc = epoch_acc / ds.GetNumBatches()
                self.stage_losses[stage].append(epoch_loss)
                self.stage_accs[stage].append(epoch_acc)
                
                if self.conf.remote_log:
                    self.run[ f"{stage}/loss"].log(epoch_loss, step=ei)
                    self.run[ f"{stage}/acc"].log(epoch_acc, step=ei)

                if stage != "valid":
                    self.best_loss[stage] = np.min([self.best_loss[stage], epoch_loss])
                    self.best_acc[stage] = np.min([self.best_acc[stage], epoch_acc])
                else:
                    if epoch_loss < self.best_loss["valid"]:
                        self.best_valid_loss_epoch = ei
                        self.best_loss["valid"] = epoch_loss

                    if epoch_acc > self.best_acc["valid"]:
                        # best_model_wts = copy.deepcopy(cnn.state_dict())
                        
                        checkp_fn = "best_rnn.onnx" if not self.dev else "dev_checkp.onnx"
                        if not conf.is_timing:
                            self.save_checkpoint(checkp_fn)
                        self.best_acc["valid"] = epoch_acc
                        self.best_valid_acc_epoch = ei
                        print(f"valid acc, best={epoch_acc:.3}, saved rnn checkpoint (epoch {ei}) at: {checkp_fn}")
                
            #< for over stages
            self.timings.append(time.perf_counter() - epoch_start)
            print(f"** epoch {ei+1}/{n_epochs}, time: {self.timings[-1]}")
            for stage in self.stages:
                print(f"\t- {stage} loss:{self.stage_losses[stage][-1]:.3}, best: {self.best_loss[stage]:.3}")
                print(f"\t- {stage} acc:{self.stage_accs[stage][-1]:.3}")

            print(f"avg timg per epoch: {np.mean(self.timings)} seconds, {H.precisedelta(np.mean(self.timings))}")
            print(f"expected training time: {H.naturaldelta(np.mean(self.timings) * (n_epochs - ei - 1))}")
            print(f"expected training time in secs: {np.mean(self.timings) * (n_epochs - ei - 1)}")
            print(f"timing of this epoch: {H.naturaldelta(self.timings[-1])} - if all like this: {H.naturaldelta(self.timings[-1]*n_epochs)}")
            
            early_stop = []
            print("- EARLY STOP, evaluating...")
            for cname, crit in self.early_stop_criteria.items():
                if type(crit) is ProgressEarlyStopping2:
                    b = crit.append(self.stage_losses["train"][-1], self.stage_losses["valid"][-1])
                else:
                    b = crit.append(self.stage_losses["valid"][-1])  # v, bool: if True then stop
                print(f"- {cname}: {b}")
                early_stop.append(b)
                self.early_stop_logs[cname].append(b)

            print(f"- EARLY STOP, stop: {np.any(early_stop)}, early break enabled? {self.conf.early_break}, epoch {ei} > {eb_kick_in}? {ei > eb_kick_in}")
            early_stop = (ei > eb_kick_in) and np.any(early_stop) and self.conf.early_break
            if early_stop:
                break

            print("")
            if self.dev and (ei == 1):
                print(f"dev mode set: breaking at epoch {ei}")
                break
        #< for over epochs
        checkp_fn = join(self.fld, "last_rnn.onnx")
        self.save_checkpoint(checkp_fn)
        print("saved model at last epoch:", checkp_fn)
        results = pd.DataFrame()
        for k, v in self.stage_losses.items():
            results[ f"{k}_loss"] = v
        for k, v in self.stage_accs.items():
            results[ f"{k}_acc"] = v

        results["optimizer"] = self.opt_name
        results["lr"] = self.conf.lr
        # results["momentum"] = self.conf.momentum
        # results["pretrained"] = self.conf.pretrained
        # results["finetune"] = self.conf.finetune
        results["folder"] = self.fld

        outfn = "rnn_results" if not self.dev else "dev_fake_results"
        results.to_csv(join(self.fld, outfn + ".csv"))
        results.to_pickle(join(self.fld, outfn + ".pkl"))
        print("saved results at:", join(self.fld, outfn + ".csv and .pkl"))
        return results
    #< train
    
    def inner_forward(self, cnn_visual, cnn_semantic, texts):
        Y = Tensor.fromarray(texts)
        Y = Tensor.onehot(Y, self.voc_size)

        for i in range(1, Y.shape[1]):
            print("up to word at index:", i)        
            
            eddl.forward([cnn_visual, cnn_semantic, word_i])




    def test(self):
        # NOT IMPLEMENTED, use predict
        pass
    #<

    def get_network(self):
        return self.rnn
    #<

    def save_checkpoint(self, filename="rec_checkpoint"):
        if filename.endswith(".onnx"):
            filename = filename[:-len(".onnx")]
        elif filename.endswith(".bin"):
            filename = filename[:-len(".bin")]
        
        #rnn = self.get_network();
        eddl.save(self.rnn, join(self.fld, filename + ".bin"))
        eddl.save_net_to_onnx_file(self.rnn, join(self.fld, filename + ".onnx"))
        print(f"saved checkpoint for the recurrent model bin|onnx format: {filename}")
    #<

    def save(self, filename="recurrent_module"):
        filename = self.prefix+filename
        if filename.endswith(".onnx"):
            filename = filename[:-len(".onnnx")]
        elif filename.endswith(".bin"):
            filename = filename[:-len(".bin")]
    
        filename = join(self.conf.exp_fld, filename)
        eddl.save_net_to_onnx_file(self.get_network(), filename + ".onnx")
        #bin_out_fn = self.conf.out_fn.replace(".onnx", ".bin")
        eddl.save(self.get_network(), filename + ".bin")
        print(f"trained recurrent model saved at: {filename} [bin onnx]")
        return filename
    #<

    # uses a non-recurrent model for the predictions
    def predict(self, stage="test"):
        self.rnn2 = self.build_model(for_predictions=True)
        rnn = self.rnn2
        
        cnn = self.cnn
        #> test on CPU (ISSUE related to eddl.getStates(.) when running on GPU)
        eddl.toCPU(cnn) 
        eddl.toCPU(rnn)
        eddl.toCPU(self.rnn)
        #<

        dl = self.dataloader
        dl.SetSplit(ecvl.SplitType.test)
        n_test_batches = dl.GetNumBatches()

        #> copy parameters from the trained recurrent network (see recurrent_models.py for layer names)
        layers_to_copy = [
            #"visual_features", "dense_alpha_v",
             "co_attention",  # "dense_alpha_s", "semantic_features" removed: not it is an input
            "lstm_cell", "out_dense", "word_embs"
        ]
        for l in layers_to_copy:
            eddl.copyParam(eddl.getLayer(self.rnn, l), eddl.getLayer(rnn, l))
        #<
        
        #> save the model for predictions
        fn = self.conf.out_fn
        onnx_fn = fn.replace(".onnx", "_pred.onnx")
        bin_fn = onnx_fn.replace(".onnx", ".bin")
        eddl.save_net_to_onnx_file(rnn, onnx_fn)
        eddl.save(rnn, bin_fn)
        print(f"recurrent model used for predictions saved at:")
        print(f"\t - onnx: {onnx_fn}")
        print(f"\t - bin: {bin_fn}")

        #> connection cnn -> rnn
        # image_in = eddl.getLayer(cnn, "input") 
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        #<

        eddl.set_mode(rnn, mode=0)
        ds = self.ds
        ds.set_stage(stage)
        dev = self.conf.dev
        n_tokens = self.conf.n_tokens

        #>
        # batch size, can be set to 1 for clarity
        # print(f"1: {len(ds)}")
        # ds.batch_size = 1
        # print(f"2: {len(ds)}")
        ds.last_batch = "drop"
        # bs = ds.batch_size
        bs = self.conf["batch_size"]
        print(f"text generation on {stage}, using batches of size: {bs}")
        #< 
        
        #> for over test dataset
        bleu = 0
        generated_word_idxs = np.zeros( (bs * len(ds), n_tokens), dtype=int)
        t1 = time.perf_counter()
        dl.Start()
        for i in range(n_test_batches):
            I, X, Y = dl.GetBatch()
            # image_ids = [os.path.basename(sample.location_[0]) for sample in I]
            image_ids = [sample.location_[0] for sample in I]
            texts = self.img_ds.loc[image_ids, "collated"]
            
            texts = np.array(texts.tolist())
            current_bs = texts.shape[0]
            #> cnn forward
            
            eddl.forward(cnn, [X])
            cnn_semantic = eddl.getOutput(cnn_out)
            cnn_visual = eddl.getOutput(cnn_top)
            thresholded = self.apply_thresholds(cnn_semantic)
            thresholded = Tensor.fromarray(thresholded)
            #<
            if dev: 
                print(f"batch, images: {X.shape}")
                print(f"\t- output. semantic: {cnn_semantic.shape}, visual: {cnn_visual.shape}")
                     
            batch_gen = \
                generate_text(rnn, n_tokens, visual_batch=cnn_visual, semantic_batch=thresholded, dev=False)
            
            generated_word_idxs[i*bs:i*bs+current_bs, :] = batch_gen
            # measure bleu
            bleu += compute_bleu(batch_gen, texts)
            # if dev:
            #     for i in range(images.shape[0]):
            #         print(f"*** batch {i+1} / {len(ds)} gen word idxs ***")
            #         print(batch_gen[i, :])
            if dev:
                break
        dl.Stop()
        #< for i over batches in the test dataset
        t2 = time.perf_counter()
        print(f"text generation on {stage} in {H.precisedelta(t2-t1)}")
        bleu = bleu / len(ds)
        if self.conf.remote_log:
            self.run[f"{stage}/bleu"] = bleu
            self.run[f"{stage}/time"] = t2 - t1

        rnn = None
        self.rnn2 = None
        gc.collect()

        if self.conf.eddl_cs == "gpu":
            print("moving modules back to GPU")
            eddl.toGPU(self.rnn)
            eddl.toGPU(self.cnn)

        return bleu, generated_word_idxs
    #< predict

        # uses a non-recurrent model for the predictions
    def predict_old(self, stage="test"):
        self.rnn2 = self.build_model(for_predictions=True)
        rnn = self.rnn2
        
        cnn = self.cnn
        #> test on CPU (ISSUE related to eddl.getStates(.) when running on GPU)
        eddl.toCPU(cnn) 
        eddl.toCPU(rnn)
        eddl.toCPU(self.rnn)
        #<
    
        #> copy parameters from the trained recurrent network (see recurrent_models.py for layer names)
        layers_to_copy = [
            "visual_features", "dense_alpha_v",
             "co_attention",  # "dense_alpha_s", "semantic_features" removed: not it is an input
            "lstm_cell", "out_dense", "word_embs"
        ]
        for l in layers_to_copy:
            eddl.copyParam(eddl.getLayer(self.rnn, l), eddl.getLayer(rnn, l))
        #<
        
        #> save the model for predictions
        fn = self.conf.out_fn
        onnx_fn = fn.replace(".onnx", "_pred.onnx")
        bin_fn = onnx_fn.replace(".onnx", ".bin")
        eddl.save_net_to_onnx_file(rnn, onnx_fn)
        eddl.save(rnn, bin_fn)
        print(f"recurrent model used for predictions saved at:")
        print(f"\t - onnx: {onnx_fn}")
        print(f"\t - bin: {bin_fn}")

        #> connection cnn -> rnn
        # image_in = eddl.getLayer(cnn, "input") 
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        #<

        eddl.set_mode(rnn, mode=0)
        ds = self.ds
        ds.set_stage(stage)
        dev = self.conf.dev
        n_tokens = self.conf.n_tokens

        #>
        # batch size, can be set to 1 for clarity
        # print(f"1: {len(ds)}")
        # ds.batch_size = 1
        # print(f"2: {len(ds)}")
        ds.last_batch = "drop"
        bs = ds.batch_size
        print(f"text generation on {stage}, using batches of size: {bs}")
        #< 
        
        #> for over test dataset
        bleu = 0
        generated_word_idxs = np.zeros( (bs * len(ds), n_tokens), dtype=int)
        t1 = time.perf_counter()
        for i in range(len(ds)):
            images, _, texts = ds[i]
            #> cnn forward
            X = Tensor(images)
            eddl.forward(cnn, [X])
            cnn_semantic = eddl.getOutput(cnn_out)
            cnn_visual = eddl.getOutput(cnn_top)
            thresholded = self.apply_thresholds(cnn_semantic)
            thresholded = Tensor.fromarray(thresholded)
            #<
            if dev: 
                print(f"batch, images: {X.shape}")
                print(f"\t- output. semantic: {cnn_semantic.shape}, visual: {cnn_visual.shape}")
                     
            batch_gen = \
                generate_text(rnn, n_tokens, visual_batch=cnn_visual, semantic_batch=thresholded, dev=False)
            
            generated_word_idxs[i*bs:i*bs+images.shape[0], :] = batch_gen
            # measure bleu
            bleu += compute_bleu(batch_gen, texts)
            # if dev:
            #     for i in range(images.shape[0]):
            #         print(f"*** batch {i+1} / {len(ds)} gen word idxs ***")
            #         print(batch_gen[i, :])
            if dev:
                break
        #< for i over batches in the test dataset
        t2 = time.perf_counter()
        print(f"text generation on {stage} in {H.precisedelta(t2-t1)}")
        bleu = bleu / len(ds)
        if self.conf.remote_log:
            self.run[f"{stage}/bleu"] = bleu
            self.run[f"{stage}/time"] = t2 - t1

        rnn = None
        self.rnn2 = None
        gc.collect()

        if self.conf.eddl_cs == "gpu":
            print("moving modules back to GPU")
            eddl.toGPU(self.rnn)
            eddl.toGPU(self.cnn)

        return bleu, generated_word_idxs
    #< predict_old
#< class