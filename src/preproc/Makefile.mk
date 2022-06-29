#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#
# MAKEFILE FOR PREPROCESSING THE IU-CHEST DATASET
#

THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile: ${THIS_MAKEFILE})

PYTHON = python3

IU_CHEST_HOME=/mnt/datasets/uc5/std-dataset
IU_CHEST_META=/mnt/datasets/uc5/meta/eddl/iuchest


# STEP 0, DOWNLOAD DATA
$(IU_CHEST_HOME)/NLMCXR_png.tgz: 
	cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz 

$(IU_CHEST_HOME)/NLMCXR_reports.tgz:
	cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

$(IU_CHEST_HOME)/text: $(BASE_DS_FLD)/NLMCXR_reports.tgz
	cd $(BASE_DS_FLD) && tar xf NLMCXR_reports.tgz && mv ecgen-radiology text

$(IU_CHEST_HOME)/image: $(BASE_DS_FLD)/NLMCXR_png.tgz
	cd $(BASE_DS_FLD) && mkdir image_ && mv NLMCXR_png.tgz image_ && cd image_ && tar xf NLMCXR_png.tgz && mv NLMCXR_png.tgz .. && cd .. && mv image_ image

download: | $(IU_CHEST_HOME)/text $(IU_CHEST_HOME)/image

# STEP 1, PREPROC
$(IU_CHEST_META)/reports_raw.pkl: 0_iuchest_preproc.py
	$(PYTHON) 0_iuchest_preproc.py \
		--ds_home=$(IU_CHEST_HOME) \
		--out_fld=$(IU_CHEST_META)

PREPROC_IUCHEST: $(IU_CHEST_META)/reports_raw.pkl

# STEP 2, CLEAN 
$(IU_CHEST_META)/reports_raw2.pkl: PREPROC_IUCHEST 1_iuchest_clean.py
	$(PYTHON) 1_iuchest_clean.py \
		--ds_home=$(IU_CHEST_HOME) \
		--meta_fld=$(IU_CHEST_META)

CLEAN_IUCHEST: $(IU_CHEST_META)/reports_raw2.pkl

# STEP 3, ENCODE

ENC_WITNESS_IU=$(IU_CHEST_META)/.encoding_witness
$(ENC_WITNESS_IU): $(IU_CHEST_META)/reports_raw2.pkl 2_iuchest_encode.py 
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) 2_iuchest_encode.py --meta_fld=$(IU_CHEST_META) --img_fld=$(IU_CHEST_HOME)/image
	@mv -f $@.tmp $@

ENCODE_IUCHEST: $(ENC_WITNESS_IU)