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
import os
from nltk import sent_tokenize
# from utils.vocabulary import Vocabulary
import re
from nltk import word_tokenize



def remove_reports_without_images(df):
    print("num images - reports with num images")
    print(df.n_images.value_counts())

    print("null major mesh", df.orig_mesh_terms.isnull().sum())
    print("empty major mesh", nnz(df.orig_mesh_terms == ""))

    # remove empty images
    iii = df["n_images"] == 0
    print("reports without images:", nnz(iii))
    df = df.drop(df[iii].index)
    print("removed reports without images, new shape", df.shape)
    return df
#< 

def all_text_to_lowercase(df):
    # all to lowercase:
    columns = ["comparison", "indication", "findings", "impression", "orig_mesh_terms", "orig_auto_term"]

    def strip_field(s):
        s = s.strip()
        if s is None or s == "":
            return s
        elif s.endswith("."):
            return s
        else:
            return s + "."

    for c in ["impression", "findings"]:
        df[c] = df[c].apply(lambda s: strip_field(s))

    def to_lowercase(s):
        if isinstance(s, str):
            return s.strip().lower()
        elif isinstance(s, list):
            return [to_lowercase(x.strip()) for x in s]
        else:
            print("errors, unexpected type:", type(s))

    for c in columns:
        df[c] = df[c].apply(to_lowercase)


    # split text into sentences

    columns = ["impression", "findings"]
    for c in columns:
        df[c + "_sents"] = df[c].apply(lambda s: sent_tokenize(s))

    for c in columns:
        df["len_" + c] = df[c].apply(lambda s: len(s.strip()))

    for c in columns:
        df["nsents_" + c] = df[c + "_sents"].apply(lambda l: len(l))

    for c in columns:
        print("column:", c)
        print(df["nsents_" + c].value_counts())

    # some reports have 0 sentences either in "findings" or "impression"

    def concat_columns(row):
        return row["findings"].strip() + " " + row["impression"].strip()

    def concat_columns2(row):
        f = row["findings"].strip()
        i = row["impression"].strip()
        if len(i) > 0 and len(f) > 0:
            return f + " " + i
        elif len(i) > 0:
            return i
        elif len(f) > 0:
            return f
        else:
            return ""


    df["raw_text"] = df.apply(concat_columns2, axis=1)
    df["len_raw_text"] = df.raw_text.apply(len)
    df["nsents_raw_text"] = df["nsents_findings"] + df["nsents_impression"]

    assert (df.nsents_raw_text == df.nsents_findings + df.nsents_impression).all()
    print(df.T)

    iii = df["raw_text"].str.len == 0
    print("empty text:", nnz(iii))

    iii = df["len_raw_text"] == 0
    print("empty text:", nnz(iii))

    iii = df["raw_text"].str == "."
    print("dot:", nnz(iii))

    print("text to lowercase, done")
    return df


def clean_mesh_terms(df):
    def clean_terms(terms):
        # e.g., terms = [ "major/minor", "major/minor/minor, ...""]
        if isinstance(terms, list):
            terms = [x.split("/")[0].strip().lower() for x in terms]  # take first element in group t1/t2/t3, make it lowercase
        elif isinstance(terms, str):
            terms = [terms.strip().lower()]
        else:
            assert False, "unexpected type: " + str(type(terms))

        # some terms contain a comma: keep only the first
        new_terms = set()
        for t in terms:
            new_terms.add(t)
        new_terms = list(new_terms)
        return new_terms
        
    #
    df["major_mesh"] = df["orig_mesh_terms"].apply(lambda x: clean_terms(x))
    df["n_major_mesh"] = df["major_mesh"].apply(lambda l: len(l))
    return df


def clean_auto_terms(df):
    # clean auto terms
    # normal when set of tags is empty
    df["orig_auto_term"] = df["orig_auto_term"].apply(lambda x: ["normal"] if len(x)==0 else x)
    list_of_auto = df.orig_auto_term.tolist()
    u_auto = set()
    for l in list_of_auto:
        for ll in l:
            u_auto.add(ll)

    print("n unique auto terms:", len(u_auto))

    # normalize terms
    norm_terms = {}
    with open("auto_term_norm.txt", "r", encoding="utf-8") as fin:
            lines = [line for line in fin.readlines() if len(line.strip()) > 0]

    for line in lines:
        subst = [t.strip() for t in line.split(":") if len(t.strip()) > 0]
        norm_terms[subst[0]] = subst[1]
        # print(f"{subst[0]} -> {subst[1]}")
    print(f"{len(norm_terms)} substitutions")

    def perform_subst(terms):
            new_terms = set()
            for t in terms:
                new_term = norm_terms.get(t, t)
                if new_term != t:
                    # print(f"{t} -> {new_term}")
                    pass
                new_terms.add(new_term)
            return sorted(list(new_terms))

    df["auto_term"] = df["orig_auto_term"].apply(lambda x: perform_subst(x))
    df["n_auto_term"] = df["auto_term"].apply(lambda l: len(l))
    return df


def clean_text_v1(text, verbose=False):
    def subst_numbers(token):
        s = re.sub(r"\A\d+(,|\.)\d+", "_NUM_", token)  # _DEC_ for finer texts
        s = re.sub(r"\A\d+", "_NUM_", s)
        return s

    def subst_meas(text):
        # substitute measures
        e = r"(_NUM_|_DEC_)\s?(cm|mm|in|xxxx)|_NUM_ x _MEAS_|_DEC_ x _MEAS_|_MEAS_ x _MEAS_ x _MEAS|_MEAS_ x _MEAS_"
        t1 = text
        while True:
            t2 = re.sub(e, "_MEAS_", t1)
            if t1 == t2:
                break
            else:
                t1 = t2
        return t1

    text2 = text.replace(" ", " ")
    text2 = text2.replace("..", ".")


    symbols = ",;:?)(!"

    e = "|".join([re.escape(s) for s in symbols])
    text2 = re.sub(e, " ", text2)
    # text2 = " ".join( [t.strip() for t in text2.split(" ")])
    # numbered list items
    text2 = re.sub(r"\s\d+\. ", " ", text2)
    # dash
    text2 = re.sub(r"-", "_", text2)
    # percentages
    text2 = re.sub(r"\d+%\s", "_PERC_ ", text2)
    # XXXX XXXX -> XXXX_XXX
    text2 = re.sub(r"xxxx(\sxxxx)+", "xxxx", text2)
    # ordinals
    text2 = re.sub(r"1st|2nd|3rd|[0-9]+th ", "_N_TH_ ", text2)


    sentences = []
    for sent in sent_tokenize(text2):
        new_tokens = [subst_numbers(token) for token in word_tokenize(sent)[:-1]]  # [:-1] not using last dot
        # for token in word_tokenize(sent):
        #     w = subst_numbers(token)
        #     new_tokens.append(w)
    
        sent = " ".join(new_tokens)
        sent = subst_meas(sent)
        sentences.append(sent)

    text2 = ". ".join(sentences) + "."  # dots, and in particular the last ., were not removed by word_tokenize

    if verbose and text != text2:   # and "_MEAS_" in text2:
        print("* IN (it has been modified):")
        print(text)
        print("* OUT:")
        print(text2)
        print(10 * "*** ")

    return text2


def remove_tags(tags, dataf):
    print("removing columns:", tags)
    def remove_tag(tag, list_of_tags):
        if tag in list_of_tags:
            list_of_tags.remove(tag)
        return list_of_tags

    for tag in tags:
        dataf.major_mesh.apply(lambda mesh: remove_tag(tag, mesh))
    
    eee = dataf.major_mesh.apply(lambda mesh: len(mesh) == 0)
    print(f"after removing tags, {nnz(eee)} reports do not have any tag, removing")
    rows1 = dataf.shape[0]
    dataf = dataf.drop(dataf[eee].index)
    rows2 = dataf.shape[0]
    print("removed rows:", rows1 - rows2)
    assert (rows1 - rows2) == nnz(eee)
    return dataf

def main(ds_home, meta_fld):
    df = pd.read_pickle( join(meta_fld, "reports_raw.pkl"))
    print(f"loaded dataset with shape: {df.shape}")

    df = remove_reports_without_images(df)
    df = all_text_to_lowercase(df)

    print(df.head().T)

    # check if there are some rows with empty text
    eee = df.raw_text.str.len() == 0
    n_empty_text = nnz(eee)
    n_empty_impression = nnz(df.impression.str.len() == 0)
    n_empty_findings = nnz(df.findings.str.len() == 0)
    print("n rows with empty raw text:", n_empty_text)
    print("n rows with empty impression:", n_empty_impression)
    print("n rows with empty findings:", n_empty_findings)
    print("removing rows with empty text")
    nrows1 = df.shape[0]
    df.drop(df.loc[eee].index, inplace=True)
    nrows2 = df.shape[0]
    print(f"removed {nrows1 - nrows2} rows because they had empty text")

    df = clean_mesh_terms(df)
    print(df["n_major_mesh"].value_counts())

    df = remove_tags(tags = ["technical quality of image unsatisfactory", "no indexing"], dataf=df)


    df = clean_auto_terms(df)
    print(df["n_auto_term"].value_counts())

    print("cleaning text... ", end="", flush=True)
    df["text"] = df.raw_text.apply(lambda text: clean_text_v1(text, verbose=False))
    print("done with cleaning")

    # remove rows with empty text
    df.to_pickle(join(meta_fld, "reports_raw2.pkl"))
    print(df.head().T)
    print(f"saved {join(meta_fld, 'reports_raw2.pkl')}")
    print("all done")

if __name__ == "__main__":
    fire.Fire(main)