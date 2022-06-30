#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#
# EXPERIMENTS WITH THE IU-CHEST DATASET
#

THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile: ${THIS_MAKEFILE})

SEED = 10
SHUFFLE_SEED = 40

BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

OUT_FLD=/mnt/datasets/uc5/EXPS/eddl
MIMIC_OUT_FLD=/mnt/datasets/uc5/EXPS/eddl/mimic
PYTHON = python3


# --------------------------------------

# Mesh, threshold 130
train_cnn_mesh_130th:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/mesh_130th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--description="mesh, th 130, resnet, balanced on normal, ES: patience" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--n_epochs=500 \
		--bs=128 \
		--finetune=True \
		--early_break=True \
		--remote_log=True \
		--nodev


train_rnn_mesh_130th:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/mesh_130th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5000 \
		--description="RNN, dropout0.5, no sem th, mesh, th 130, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--remote_log=True \
		--semantic_thresholds=False \
		--dropout=0.5 \
		--eb_from_epoch=400 \
		--nodev

test_rnn_mesh_130th:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/mesh_130th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev

# --------------------------------------


# Mesh, threshold 130
train_cnn_mesh_130th_noft:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/mesh_130th_noft --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="mesh, no finetune, th 130, resnet, balanced on normal, ES: patience" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--n_epochs=500 \
		--finetune=False \
		--early_break=True \
		--remote_log=True \
		--nodev



# --------------------------------------

train_cnn_mesh_70th:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/mesh_70th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="mesh, th 70, resnet, balanced on normal, ES: patience" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--n_epochs=500 \
		--eddl_cs=gpu \
		--bs=128 \
		--finetune=True \
		--early_break=True \
		--remote_log=True \
		--nodev

train_rnn_mesh_70th:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/mesh_70th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5000 \
		--description="RNN, no sem th, mesh, th 70, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=False \
		--remote_log=True \
		--semantic_thresholds=False \
		--nodev


test_rnn_mesh_70th:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/mesh_70th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev

# --------------------------------------



train_cnn_mesh_70th_noft:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/mesh_70th_noft --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="mesh, no finetune, th 70, resnet, balanced on normal, ES: patience" \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--n_epochs=1000 \
		--eddl_cs=gpu \
		--bs=128 \
		--finetune=False \
		--early_break=True \
		--remote_log=True \
		--nodev


# -----------------------------------------------------------------------------


train_cnn_auto_70th_sl18:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/auto_50th_seqlen18 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="auto, th 50, seqlen 18, resnet, balanced on normal, early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--finetune=True \
		--early_break=True \
		--remote-log=True \
		--nodev

test_cnn_auto_70th_sl18:
	$(PYTHON) 2_test_cnn.py $(OUT_FLD)/auto_50th_seqlen18 \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=256 \
		--nodev

train_rnn_auto_70th_sl18:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_seqlen18 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=2000 \
		--description="RNN, auto, seqlen 18, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--eddl_cs_mem=low_mem \
		--bs=128 \
		--early_break=True \
		--eb_from_epoch=1000 \
		--remote_log=True \
		--nodev

test_rnn_auto_70th_sl18:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th_seqlen18 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev

# -----------------------------------------------------------------------------


train_cnn_auto_50th:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/auto_50th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=64 \
		--finetune=True \
		--early_break=True \
		--remote-log=False \
		--nodev

test_cnn_auto_50th:
	$(PYTHON) 2_test_cnn.py $(OUT_FLD)/auto_50th \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,1] \
		--eddl_cs=gpu \
		--bs=256 \
		--nodev

train_rnn_auto_50th:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=2000 \
		--description="RNN, auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=False \
		--remote_log=True \
		--nodev

test_rnn_auto_50th:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev
# ------------------------------------------------------------------------------------------------
# DROPOUT
# -----------------------------------------------------------------------------


train_cnn_auto_50th_dropout:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=64 \
		--finetune=True \
		--early_break=True \
		--remote-log=False \
		--nodev

test_cnn_auto_50th_dropout:
	$(PYTHON) 2_test_cnn.py $(OUT_FLD)/auto_50th_dropout \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,1] \
		--eddl_cs=gpu \
		--bs=256 \
		--nodev

train_rnn_auto_50th_dropout:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=1000 \
		--description="RNN, dropout, auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--eddl_cs_mem=full_mem \
		--bs=128 \
		--early_break=True \
		--eb_from_epoch=500 \
		--remote_log=True \
		--dropout=0.5 \
		--nodev

test_rnn_auto_50th_dropout:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev

test_rnn_demo:
	$(PYTHON) 4_test_rnn.py --exp_fld=/opt/uc5/results/demo/auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--nodev



# ----
train_cnn_auto50_seed23:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/auto_50th_seed23 --seed=99 --shuffle_seed=88 \
		--n_epochs=500 \
		--description="CNN, auto, th 50, seed 23, resnet, balanced on normal, early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=256 \
		--finetune=True \
		--early_break=True \
		--eb_from_epoch=50 \
		--remote-log=True \
		--nodev

# tmux 0 - 2022-06-22
train_rnn_auto50_seed23:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_seed23 --seed=99 --shuffle_seed=88 \
		--n_epochs=1300 \
		--description="LSTM, seed 23, diff split, CNN trained with FINETUNE, AUTO th 50, dropout 0.2, std split" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--eb_from_epoch=900 \
		--remote_log=True \
		--semantic_thresholds=False \
		--dropout=0.3 \
		--nodev

# >>>>>>>>>>>>>>>>>>

train_rnn_auto50_do03:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_do03 --seed=99 --shuffle_seed=88 \
		--n_epochs=1300 \
		--description="LSTM, FNN trained with FINETUNE, AUTO th 50, dropout 0.3, std split" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--eb_from_epoch=900 \
		--remote_log=True \
		--semantic_thresholds=False \
		--early_break=False \
		--nodev

test_rnn_auto50_do03:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th_do03 --seed=99 --shuffle_seed=88 \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--stages=[test] \
		--nodev

# <<<<<<<<<<<<<<<<<<<<


train_rnn_auto50_do03_seed23:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_seed23 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=2000 \
		--description="RNN, thresholds, auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=False \
		--remote_log=True \
		--semantic_thresholds=False \
		--early_break=False \
		--nodev

test_rnn_auto50_do03_seed23:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th_seed23 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev
# ----

# ----
train_rnn_auto_50th_sem_th:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/auto_50th_sem_th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5000 \
		--description="RNN, thresholds, auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=False \
		--remote_log=True \
		--semantic_thresholds=True \
		--nodev

test_rnn_auto_50th_sem_th:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/auto_50th_sem_th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--dev
# ----
# test timings for deliverable
test_timings_auto_terms:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/test_timings --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="CNN, test timings" \
		--dataset_name=chest-iu \
		--gpu_id=[1,1,1,1] \
		--eddl_cs=gpu \
		--eddl_cs_mem=mid_mem \
		--bs=256 \
		--finetune=True \
		--early_break=False \
		--remote-log=False \
		--is_timing=True \
		--nodev


test_timings_auto_terms_rnn:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/test_timings --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="RNN, test timings" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,1] \
		--eddl_cs=gpu \
		--eddl_cs_mem=mid_mem \
		--bs=128 \
		--finetune=False \
		--early_break=False \
		--remote-log=False \
		--is_timing=True \
		--nodev

train_cnn_test:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/test_auto_50th --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=50 \
		--description="auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,0,1] \
		--eddl_cs=gpu \
		--bs=64 \
		--finetune=True \
		--early_break=True \
		--remote-log=False
		--nodev


# ********************

# *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
# ATTENTION

train_rnn_attn:
	$(PYTHON) 6_train_rnn_attn.py $(OUT_FLD)/attn_sl12 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5000 \
		--description="RNN with attention, auto terms" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--remote_log=False \
		--semantic_thresholds=False \
		--dev



# ---
# GRU

train_gru:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/gru_test_auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=1000 \
		--description="GRU, auto terms, threshold 50, dropout 0.2" \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--remote_log=True \
		--dropout=0.2 \
		--semantic_thresholds=False \
		--rec_cell_type=gru \
		--nodev

train_gru_2:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/gru_test_2_auto_50th_do03 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=1200 \
		--description="GRU, auto terms, threshold 50, dropout 0.3" \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--early_break=True \
		--remote_log=True \
		--dropout=0.3 \
		--semantic_thresholds=False \
		--rec_cell_type=gru \
		--nodev

test_gru:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/gru_test_auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--rec_cell_type=gru \
		--stages=["test"] \
		--nodev


test_gru_2:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/gru_test_2_auto_50th_do03 --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--rec_cell_type=gru \
		--stages=["test"] \
		--nodev


# --------------------------

# TEST FOR THE DEMO

final_train_cnn:
	$(PYTHON) 1_train_cnn.py $(OUT_FLD)/final_exp --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=2 \
		--description="TARGET: final_train_cnn - final, demo" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--finetune=True \
		--early_break=True \
		--remote-log=True \
		--nodev

final_test_cnn:
	$(PYTHON) 2_test_cnn.py $(OUT_FLD)/final_exp \
		--dataset_name=chest-iu \
		--gpu_id=[0,0,1,1] \
		--eddl_cs=gpu \
		--bs=128 \
		--nodev

final_train_rnn:
	$(PYTHON) 3_train_rnn.py $(OUT_FLD)/final_exp --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=2 \
		--description="TARGET: final_train_rnn - RNN, dropout, auto, th 50, resnet, balanced on normal, no early break, finetune" \
		--dataset_name=chest-iu \
		--gpu_id=[1,0,0,0] \
		--eddl_cs=gpu \
		--eddl_cs_mem=mid_mem \
		--bs=128 \
		--early_break=True \
		--eb_from_epoch=2 \
		--remote_log=True \
		--dropout=0.2 \
		--nodev

final_test_rnn:
	$(PYTHON) 4_test_rnn.py --exp_fld=$(OUT_FLD)/final_exp --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--nodev


# *** FULL TEST FOR DEMO
test_rnn_demo:
	$(PYTHON) 4_test_rnn.py --exp_fld=/opt/uc5/results/demo/auto_50th_dropout --seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--dataset_name=chest-iu \
		--gpu_id=[0,1,0,0] \
		--eddl_cs=gpu \
		--bs=128 \
		--nodev

