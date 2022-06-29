#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import fire
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
import os
import numpy as np
from numpy import count_nonzero as nnz
import pandas as pd
import pickle
from posixpath import join
from tqdm import tqdm
from nltk import sent_tokenize

from utils.vocabulary import Vocabulary



def encode_labels(rep_ids, image_filenames, terms, images_fld):
    uterms = set()
    for term_list in terms:
        for t in term_list:
            uterms.add(t)
        #<

    print(f"unique terms: {len(uterms)}")
    uterms = ["normal"] + sorted([l for l in uterms if l != "normal"])

    matrix = []  # rows correspond to images
    img_index = []  # used with matrix
    rep_index = []
    paths = []
    rep_matrix = []  # rows correspond to reports
   
    for rep_id, filenames, term_list in zip(rep_ids, image_filenames, terms):
        # print("*")
        # print("id:", rep_id)
        # print("image_filenames:", filenames)
        # print("terms:", term_list)
        enc = []
        for term in uterms:
            enc.append(term in term_list)
        rep_matrix.append(enc)
        rep_index.append(rep_id)

        for fn in filenames:
            img_index.append(fn)
            paths.append(join(images_fld, fn))
            matrix.append(enc)

    rep_ds = pd.DataFrame(data=np.array(rep_matrix).astype(int), columns=uterms)
    rep_ds["id"] = rep_index
    rep_ds.set_index("id", inplace=True)
    print(f"dataframe, index is report id: {rep_ds.shape}")
    # display(rep_ds.head())

    img_ds = pd.DataFrame(data=np.array(matrix).astype(int), columns=uterms)
    img_ds["filename"] = img_index
    img_ds.set_index("filename", inplace=True)
    print(f"dataframe, index is image filename: {img_ds.shape}")
    return rep_ds, img_ds

def save_pkl_csv(dataframe, meta_fld, fn):
    out_fn = join(meta_fld, fn + ".pkl")
    dataframe.to_pickle(out_fn)
    print(f"saved {out_fn}")
    out_fn = join(meta_fld, fn + ".csv")
    dataframe.to_csv(out_fn, sep="\t")
    print(f"saved` {out_fn}")

def main(meta_fld, img_fld, n_words=1000):
    df = pd.read_pickle(join(meta_fld, "reports_raw2.pkl"))

    print("ENCODING MESH TERMS")
    mesh_terms_s = df.orig_mesh_terms.str.join("/").tolist()
    df["mesh_term_s"] = mesh_terms_s
    mesh_terms_s = set(mesh_terms_s)
    print("unique combos of mesh terms:", len(mesh_terms_s))

    
    df2 = df.reset_index()
    rep_ids = df2["id"].tolist()
    image_filenames = df2["image_filename"].tolist()
    terms = df2["major_mesh"].tolist()
    rep_ds, img_ds = encode_labels(rep_ids, image_filenames, terms, img_fld)
    print(img_ds.T)
    save_pkl_csv(rep_ds, meta_fld, "rep_dataset")
    save_pkl_csv(img_ds, meta_fld, "img_dataset")
    
    auto_term_s = df.auto_term.str.join("/").tolist()
    df["auto_term_s"] = auto_term_s
    auto_term_s = set(auto_term_s)
    print("unique combos of auto terms:", len(auto_term_s))
    terms = df2["auto_term"].tolist()
    rep_ds, img_ds = encode_labels(rep_ids, image_filenames, terms, img_fld)
    print(img_ds.T)
    save_pkl_csv(rep_ds, meta_fld, "rep_dataset_auto")
    save_pkl_csv(img_ds, meta_fld, "img_dataset_auto")
    print("done with image labels")

    # vocabulary
    text_col = df.text
    vocab = Vocabulary()
    for (id, text) in text_col.iteritems():
        for sentence in sent_tokenize(text):
            if len(sentence) == 0:
                print("ERROR, sentence length == 0")
            vocab.add_sentence(sentence)
    
    print("number of distinct words:", len(vocab.word2idx))
    print("total number of words:", vocab.word_count)
    out_fn = join(meta_fld, "vocab.pkl")
    with open(out_fn, "wb") as fout:
        pickle.dump(vocab, fout)
    
    print(f"saved {out_fn}")
    
    wc = list(vocab.word2count.items())
    # words sorted according to their absolute frequency in the dataset
    wc = sorted(wc, key=lambda elem: -elem[1])
    n = 0
    for w, c in wc:
        n += c
    print("(all) total number of words (summed):", n)

    wc2 = wc[:n_words]

    n2 = 0
    for w, c in wc2:
        # print(f"{w}:{c}")
        n2 += c

    # for word, count in vocab.word2count.items():
    #     print(f"{word}: {count}")

    print("(all) n:", n)
    print("(filt) n2:", n2)

    print("coverage after filtering:", n2 / n)
    print("diff in word count:", n2 - n)
    vocab.keep_n_words(n_words)
    print("saved word count:", vocab.word_count)
    out_vocab_fn = join(meta_fld,  f"vocab_{n_words}.pkl")
    with open(out_vocab_fn, "wb") as fout:
        pickle.dump(vocab, fout)
    print(f"saved {out_vocab_fn}")

    print("encoding, all done.")

    
if __name__ == "__main__":
    fire.Fire(main)