#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#
# EXPERIMENTS WITH THE MIMIC-CXR DATASET 

THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running MIMIC makefile: ${THIS_MAKEFILE})

SEED = 10
SHUFFLE_SEED = 40

BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

OUT_FLD=/mnt/datasets/uc5/EXPS/mimic
PYTHON = python3

# Mesh, threshold 130
train_cnn:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/std_exp_resized_2 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5 \
		--description="mimic, cnn, early break" \
		--dataset_name=mimic_cxr \
		--gpu_id=[0,0,0,1] \
		--n_epochs=1000 \
		--eddl_cs=gpu \
		--eddl_cs_mem=mid_mem \
		--bs=256 \
		--finetune=True \
		--early_break=True \
		--remote_log=True \
		--nodev

train_rnn:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/std_exp_resized_2 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=500 \
		--description="mimic, early break, dropout 0.5" \
		--dataset_name=mimic_cxr \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=256 \
		--early_break=True \
		--eb_from_epoch=130 \
		--remote_log=True \
		--semantic_thresholds=False \
		--one_param_only=False \
		--dropout=0.5 \
		--nodev



test_rnn:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/std_exp_resized --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev


#
train_cnn_timing:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/std_exp_resized_timing --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5 \
		--description="mimic, cnn, early break" \
		--dataset_name=mimic_cxr \
		--gpu_id=[1,0,0,0] \
		--n_epochs=1000 \
		--eddl_cs=gpu \
		--eddl_cs_mem=mid_mem \
		--bs=256 \
		--finetune=True \
		--early_break=True \
		--remote_log=False \
		--is_timing=True \
		--nodev

train_rnn_timing:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/std_exp_resized_timing --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=500 \
		--description="mimic, early break, dropout 0.5" \
		--dataset_name=mimic_cxr \
		--gpu_id=[1,1,0,0] \
		--eddl_cs=gpu \
		--bs=256 \
		--early_break=True \
		--eb_from_epoch=130 \
		--remote_log=False \
		--semantic_thresholds=False \
		--one_param_only=False \
		--dropout=0.5 \
		--is_timing=True \
		--nodev


train_rnn_v2:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/std_exp_resized_cloned --seed=535 --shuffle_seed=434 \
		--n_epochs=100 \
		--description="mimic, early break after 50e, lr 1e-5" \
		--dataset_name=mimic_cxr \
		--gpu_id=[0,0,1,1] \
		--eddl_cs=gpu \
		--bs=256 \
		--lr=1e-5 \
		--early_break=True \
		--eb_from_epoch=50 \
		--remote_log=True \
		--semantic_thresholds=False \
		--one_param_only=False \
		--nodev


# ********************
# MIMIC

