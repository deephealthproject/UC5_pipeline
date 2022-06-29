#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import humanize as H
import numpy as np
import pandas as pd
from posixpath import join
import time
from tqdm import tqdm

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor



from pyeddl._core import Loss, Metric
from training.early_stopping import UpEarlyStopping, GLEarlyStopping, ProgressEarlyStopping2, PatienceEarlyStopping

class Jaccard(Metric):
    def __init__(self):
        Metric.__init__(self, "py_jaccard")

    def value(self, t, y):
        t.info()
        y.info()
        n_labs = t.sum()
        #print(f"n labels {n_labs}", flush=True)
        y_round = y.round()
        #print(f"predicted: {y.round().sum()}", flush=True)
        score = t.mult(y_round).sum()
        #print(f"correctly predicted: {score}", flush=True)
        return score / n_labs


# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)
    

class EddlCnnModule_ecvl:
    def __init__(self, dataset, opt_name, lr, pretrained, finetune, momentum, config, neptune_run=None, name=""):
        self.ds = dataset
        self.n_classes = len(self.ds.classes_)
        print("number of classes:", self.n_classes)
        
        print(f"number of classes (output layer): {self.n_classes}")

        self.conf = Bunch(**config)
        self.opt_name = opt_name
        self.lr = lr
        self.pretrained = pretrained
        self.finetune = finetune
        self.momentum = momentum
        print(f"number of classes (output layer): {self.n_classes}")
        self.conf = Bunch(**config)
        self.name = name
        self.out_layer_act = "sigmoid"
        self.verbose = self.conf.verbose
        self.img_size = self.conf.img_size
        self.out_fld = self.conf.out_fld

        if ("load_file" in self.conf) and (self.conf.load_file is not None):
            print(f"loading model from file {self.conf.load_file}")
            self.cnn = self.load_model()
        else:
            self.cnn = self.build_cnn()
        
        self.run = neptune_run
        # if self.conf.remote_log:
        #    import neptune.new as neptune
        #    self.run = self.init_neptune(neptune_run)
        
        # set seed of the augmentation container
        self.layer_names = None
        self.dev = self.conf.dev
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
        self.best_loss = {stage: np.inf for stage in self.stages}
        self.stage_accs = { stage: [] for stage in self.stages }
        self.timings = []
        upEs = UpEarlyStopping()
        glEs = GLEarlyStopping()
        prEs = ProgressEarlyStopping2()
        patEs = PatienceEarlyStopping(patience=5, is_loss=True)
        self.early_stop_criteria = {"up": upEs, "gl":glEs, "progress":prEs }  # , "patience":patEs}
        self.early_stop_logs = { name: list() for name in self.early_stop_criteria.keys() }
    #<

    def delete_nn(self):
        del self.cnn
    #<

    def init_neptune(self, run):
        if not run:
            if self.conf["dev"]:
                neptune_mode = "debug"
            elif self.conf["remote_log"]:
                neptune_mode = "async"
            else:
                neptune_mode = "offline"
            print(f"NEPTUNE REMOTE LOG, mode set to {neptune_mode}")
            run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
        
        run["description"] = self.conf.description if "description" in self.conf else "cnn_module"
        run["configuration"] = self.conf
        run["num image classes"] = self.n_classes
        return run 
    #<
   
    def comp_serv(self, eddl_cs=None, eddl_mem=None):
        eddl_cs = eddl_cs or self.conf.eddl_cs
        eddl_mem = eddl_mem or self.conf.eddl_cs_mem

        print("creating computing service:")
        print(f"computing service: {eddl_cs}")
        print(f"memory: {eddl_mem}")

        if self.conf.bs == 32:
            lsb = 1
        else:
            lsb = 1
        # lsb = 10

        return eddl.CS_GPU(g=self.conf.gpu_id, mem=self.conf.eddl_cs_mem, lsb=lsb) if eddl_cs == 'gpu' else eddl.CS_CPU(th=1, mem=eddl_mem)
    #< 

    def get_loss_name(self):
        if self.conf.activation == "softmax":
            name = "softmax_cross_entropy"
            print("output layer:", self.out_layer_act)
        elif self.conf.activation == "sigmoid":
            name = "binary_cross_entropy"
        else:
            assert False, f"check activation of the output layer, currently selected {self.conf.activation}"
        
        print("LOSS function:", name)
        return name
    #<

    def get_optimizer(self):
        opt_name = self.opt_name  # self.conf.optimizer
        print("optimizer in configuration:", opt_name)
        if opt_name == "adam":
            return eddl.adam(lr=self.lr)
        elif opt_name == "cyclic":
            #self.cylic = 
            return eddl.adam(lr=self.lr)
        else:
            assert False
    #<

    def download_base_cnn(self, top=True):
        return eddl.download_resnet18(top=top)  #, input_shape=[1, 224, 224]) 
    #<

    # returns an output layer with the activation specified via cli
    def get_out_layer(self, top_layer, version="sigmoid", layer_name="cnn_out"):
        print(f"cnn, output layer: {version}")
        res = None
        print(f"cnn, number of classes {self.n_classes}")
        dense_layer = eddl.HeUniform(eddl.Dense(top_layer, self.n_classes, name="out_dense"))
        dense_layer.initialize()
        
        if self.out_layer_act == "sigmoid":
            print("created output layer with sigmoid activation")
            res = eddl.Sigmoid(dense_layer, name=layer_name)
        elif self.out_layer_act == "softmax":
            assert False
            res = eddl.Softmax(dense_layer, name=layer_name)
        else:
            assert False, "unsupported activation"

        return res
    #<

    def freeze_convolutional(self):
        for layer in self.layer_names:
                eddl.setTrainable(self.base_cnn, layer, False)
        self.finetune = False

    def create_cnn(self, fine_tune=False):
        remove_top_layer = True
        base_cnn = self.download_base_cnn(top=remove_top_layer)  # top=True -> remove output layer
        self.base_cnn = base_cnn

        self.layer_names = [_.name for _ in base_cnn.layers]
        if not self.finetune:
            print("**** no fine tuning ****")
        else:
            print("fine tuning all layers")
        for layer in self.layer_names:
                eddl.setTrainable(base_cnn, layer, self.finetune)

        cnn_in = eddl.getLayer(base_cnn, "input")
        cnn_top = eddl.getLayer(base_cnn, "top")
        cnn_out = self.get_out_layer(cnn_top, version=self.conf.activation)
        eddl.setTrainable(base_cnn, "cnn_out", True)

        cnn = eddl.Model([cnn_in], [cnn_out])
        return cnn
    #<

    def build_cnn(self, cnn=None):
        if cnn is None:
            cnn = self.create_cnn()
        #<
        loss_str = self.get_loss_name()
        optimizer = self.get_optimizer()
        loss = eddl.getLoss(loss_str)
       
        metric = eddl.getMetric("accuracy")  # Jaccard()
        # eddl.build(cnn, optimizer, [loss_str], ["binary_accuracy"], self.comp_serv(), init_weights=False)  # losses, metrics, 
        cnn.build(optimizer, [loss], [metric], self.comp_serv(), initialize=False)  # losses, metrics, 

        print(f"cnn built: resnet18")
        return cnn
    #<

    def load_model(self, filename=None):
        filename = filename or self.conf.load_file
        cnn = eddl.import_net_from_onnx_file(filename)
        return self.build_cnn(cnn)
    #<

    def get_network(self):
        return self.cnn
    #<

    def train(self):
        print("training is starting, epochs:", self.conf.n_epochs)
        print("exp folder:", self.conf.exp_fld)
        print("out folder:", self.conf.out_fld)
        

        ds = self.ds
        cnn = self.get_network()
        eddl.summary(cnn)
        print("- num classes:", len(ds.classes_))
        
        print("- training batches:", ds.GetNumBatches(ecvl.SplitType.training))
        print("- validation batches:", ds.GetNumBatches(ecvl.SplitType.validation))
        print("- test batches:", ds.GetNumBatches(ecvl.SplitType.test))
        print("- dev set?", self.dev)
        print("- finetune?", self.finetune)
        print("- is timing test?", self.conf.is_timing)

        ds.SetSplit(ecvl.SplitType.training)
        n_epochs = self.conf.n_epochs
        n_training_batches = ds.GetNumBatches()
        timing_batches = []

        for ei in range(n_epochs):
            
            print(f"epoch {ei+1} / {n_epochs}")
            epoch_start = time.perf_counter()
            for stage in self.stages.keys():
                if self.conf.is_timing and stage == "test":
                    self.stage_losses["test"].append(0.0)
                    self.stage_accs["test"].append(0.0)
                    print("timing mode, skipping test stage")
                    continue

                eddl.reset_loss(cnn)
                print("> STAGE:", stage)
                ds.SetSplit(self.stages[stage])
                num_batches = ds.GetNumBatches()
                ds.ResetBatch(shuffle = (stage == "train") )
                ds.Start()
                epoch_loss = 0
                epoch_acc = 0
                # batches_timings, used only during development
                batches_timings = []
                for bi in range(ds.GetNumBatches()):
                    batch_t0 = time.perf_counter()
                    # print(".", end="", flush=True)
                    # if (bi+1) % 10 == 0:
                    #     print("#", flush=True)
                    batch_t0 = time.perf_counter()
                    _, X, Y = ds.GetBatch()
                    # print(X.shape)
                    if stage == "train":
                        eddl.train_batch(cnn, [X], [Y])
                    else:                
                        eddl.eval_batch(cnn, [X], [Y])

                    loss = eddl.get_losses(cnn)[0]
                    acc = eddl.get_metrics(cnn)[0]
                    epoch_loss += loss
                    epoch_acc += acc
                    if self.dev and bi == 10:
                        print("breaking at batch 10 because dev is set")
                        break
                    
                    # if bi < 10:
                    #     batches_timings.append(time.perf_counter() - batch_t0)
                    #     print("avg time per batch:", H.precisedelta(np.mean(batches_timings)))
                    if self.conf.is_timing:
                        timing_batches.append(time.perf_counter() - batch_t0)
                        avg_t_batch = np.mean(timing_batches)
                        _stage_t = avg_t_batch * num_batches
                        print(f"batch proc in {H.precisedelta(avg_t_batch)}, {stage} in: {H.precisedelta(_stage_t)}, remaining {H.precisedelta(_stage_t - avg_t_batch * (bi+1))}")

                        if bi == 100:
                            break
                # for over batches
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
                else:
                    if epoch_loss < self.best_loss["valid"]:
                        # best_model_wts = copy.deepcopy(cnn.state_dict())
                        checkp_fn = join(self.out_fld, "best_cnn.onnx")
                        if not self.conf.is_timing:
                            self.save_checkpoint(checkp_fn)
                        self.best_loss["valid"] = epoch_loss
                        self.best_valid_loss_epoch = ei
                        print(f"valid loss, best={epoch_loss:.3}, saved cnn checkpoint (epoch {ei}) at: {checkp_fn}")
            # for over stages
            
            # print(f"epoch {ei+1}/{n_epochs}. {stage}, current loss {epoch_loss:.3f}, best {self.best_loss[stage]:.3f}")
            self.timings.append(time.perf_counter() - epoch_start)
            print(f"** epoch {ei+1}/{n_epochs}, time: {self.timings[-1]}")
            
            for stage in self.stages:
                print(f"\t- {stage} loss:{self.stage_losses[stage][-1]:.3}, best: {self.best_loss[stage]:.3}")
                # print(f"\t- {stage} acc:{stage_accs[stage][-1]:.3}")
            
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

            print(f"- EARLY STOP, stop: {np.any(early_stop)}, early break enabled? {self.conf.early_break}")
            early_stop = np.any(early_stop) and self.conf.early_break
            if early_stop:
                break
            
            print("")
            if self.dev and (ei == 1):
                print(f"dev mode set: breaking at epoch {ei}")
                break
                # check early breaking
        # for over epochs
        checkp_fn = join(self.out_fld, "last_cnn.onnx")
        self.save_checkpoint(checkp_fn)
        print("saved model at last epoch:", checkp_fn)
        results = pd.DataFrame()
        for k, v in self.stage_losses.items():
            results[ f"{k}_loss"] = v
        for k, v in self.stage_accs.items():
            results[ f"{k}_acc"] = v

        results["optimizer"] = self.opt_name
        results["lr"] = self.lr
        results["momentum"] = self.momentum
        results["pretrained"] = self.pretrained
        results["finetune"] = self.finetune
        results["folder"] = self.conf.out_fld

        results.to_csv(join(self.conf.out_fld, "results.csv"))
        results.to_pickle(join(self.conf.out_fld, "results.pkl"))
        return results

    #< train
   

    def save(self, filename):
        filename = join( self.conf.out_fld, filename )
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"model saved, location: {filename}")
        return filename
    #<
    
    def save_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.out_fld, filename)
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        eddl.save(self.get_network(), filename.replace(".onnx", ".bin"))
        print(f"saved checkpoint: {filename}")
    #<
    def load_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.out_fld, filename)
        print("loading last checkpoint")
        return self.build_cnn(self.load_model(filename))
    #<
#< class