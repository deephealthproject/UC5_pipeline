#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import numpy as np

eps = 1e-8


class PatienceEarlyStopping:
    def __init__(self, patience=5, min_epochs=None, is_loss=False):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = min_epochs or patience
        self.k = patience
        self.is_loss = is_loss
        
    def append(self, value):
        self.v.append(value)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        k = self.k
        assert k <= self.min_epochs, "wrong choice for k in evaluate_patience"

        if len(self.v) < self.min_epochs:
            return False # do not stop        

        a = np.array(self.v[-k:])
        assert len(a) == k
        a = a[1:] - a[:-1]
        if self.is_loss:
            # stop when loss[t+1] >= loss[t]  for the past k steps
            res =  np.all(a >= eps)
        else:
            # stop when metric[t+1] <= metric[t] for the past k steps
            res = np.all(a <= eps)
        print(a)
        print(res)
        return res
    #<

    def __repr__(self):
        rep = f"Patience stopping criterion, patience={self.k} epochs"
        return rep
#< class PatienceEarlyStopping


class GLEarlyStopping:
    def __init__(self, alpha=3, min_epochs=30):
        self.v = []
        self.min_v = np.inf
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = min_epochs
        self.alpha = alpha

    def append(self, value):
        self.v.append(value)
        if self.min_v > value:
            self.min_v = value
        
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)

        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.v) < self.min_epochs:
            return False
        value = 100 * (self.v[-1] / self.min_v - 1)
        return value > self.alpha

# criterio Up_i with strip k (Early Stopping — But When? Prechelt 2012)
class UpEarlyStopping:
    def __init__(self, i=4, k=5):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0  # iteration where stop becomes True
        self.min_epochs = i * k
        self.i = i  # number of consecutive epoch strips, k-epoch long
        self.k = k
    #<

    def append(self, val):  # expects a metric as a percentage 
        self.v.append(val)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()

        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.v) < self.min_epochs:
            return False # go
        elif len(self.v) % self.k != 0:
            return False
        
        i = self.i
        k = self.k
        v = np.array(self.v)
        indexes = np.arange(len(v) - i * k, len(v), k)
        # print(indexes)
        values = np.array(v[indexes])
        diffs = values[1:] - values[:-1]
        # print(diffs <= eps)
        stop = np.all(diffs >= eps)
        if stop:
            print("UP CRITERION, loss differences:", diffs)
        return stop
    
    def __repr__(self):
        rep = f"UpStoppingCriterion, strip={self.k}, i consecutive strips={self.i}"
        return rep
#< class UpEarlyStopping


# criterio Up_i with strip k (Early Stopping — But When? Prechelt 2012)
class ProgressEarlyStopping2:
    def __init__(self, k=5, alpha=0.5):
        self.train_errs = []  # values of the watched metric
        self.valid_errs = []

        self.stop = False
        self.iter_stop = 0
        self.min_epochs = 20
        self.alpha = alpha  # number of consecutive epoch strips, k-epoch long
        self.k = k 

    def append(self, train_err, valid_err):
        self.train_errs.append(train_err)
        self.valid_errs.append(valid_err)

        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.train_errs)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.train_errs) < self.min_epochs:
            return False # go
        elif len(self.train_errs) % self.k != 0:
            return False

        a = self.alpha
        k = self.k
        tr_e = self.train_errs[-k:]
        va_e = self.valid_errs[-k:]

        # first, progress (on train)
        p = ( sum(tr_e) / (k * min(tr_e)) - 1) * 1000
        # generalization
        gl = (va_e[-1] / min(va_e) - 1) * 100
        # criterion
        value = gl / p
        self.stop = value > a
        # print(f"progress, crit value: {value:.3}, threshold is {a} -> stop: {self.stop}")      
        return self.stop
        
    
    def __repr__(self):
        rep = f"UpStoppingCriterion, strip={self.k}, i consecutive strips={self.i}"
        return rep
#< class UpEarlyStopping

# criterio Up_i with strip k (Early Stopping — But When? Prechelt 2012)
class ProgressEarlyStopping:
    def __init__(self, k=5, theta=0.03):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = 20
        self.theta = theta  # number of consecutive epoch strips, k-epoch long
        self.k = k 

    def append(self, value):
        self.v.append(value)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.v) < self.min_epochs:
            return False # go
        elif len(self.v) % self.k != 0:
            return False

        theta = self.theta
        k = self.k
        values = self.v[-k:]
        
        mean = sum(values) / len(values) + eps
        m = min(values) * k + eps
        crit = 1000 * (mean / m - 1)
        
        self.stop = crit < theta
        return self.stop
        
    
    def __repr__(self):
        rep = f"UpStoppingCriterion, strip={self.k}, i consecutive strips={self.i}"
        return rep
#< class UpEarlyStopping
