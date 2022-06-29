#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

from PIL import Image
import glob
import os
from posixpath import join
from tqdm import tqdm
import time
import humanize as H
import multiprocessing as mp

def process(filename, ORI_PATH, PATH, NEW_SIZE=300):
    out_fn = filename.replace(ORI_PATH, PATH)
    #print("input:", filename)
    #print("output:", out_fn)
    out_fld = out_fn.rsplit('/', 1)[0]
    #print("out_fld:", out_fld)
    os.makedirs(out_fld, exist_ok=True)
    img = Image.open(filename).resize((NEW_SIZE,NEW_SIZE))
    # print("saving:", out_fn)
    print("saved:", out_fn)
    img.save(out_fn)

def main():
        # new folder path (may need to alter for Windows OS)
    # change path to your path
    ORI_PATH = "/mnt/datasets/mimic-cxr/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0"
    NEW_SIZE = 300
    PATH = "/mnt/datasets/mimic-cxr/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0_resized"

    # create new folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    filenames = glob.glob(ORI_PATH+'/**/*.jpg', recursive=True)
    print(len(filenames))
    pbar = tqdm(total=len(filenames))

    parallel_args = [(filename, ORI_PATH, PATH, NEW_SIZE) for filename in filenames]
    t0 = time.perf_counter()
    with mp.Pool(processes=8) as pool:
        pool.starmap(process, parallel_args)
        
    # loop over existing images and resize
    # change path to your path    
    print("time:", H.precisedelta(time.perf_counter() - t0))
    print("all done.")



if __name__ == "__main__":
    main()