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
import os
from tqdm import tqdm

def filename_from_path(path, keep_extension=True):
    base = os.path.basename(path)
    if keep_extension:
        return base

    pre, _ = os.path.splitext(base)
    return pre
    
def parse_id(soup):
    keys = ['pmcid', 'iuxrid', 'uid']
    d = defaultdict(None)
    selected_id = None
    for k in keys:
        if soup(k):
            # since: soup(k) returns:
            #        [<pmcid id="3315"></pmcid>]
            # 1) soup(k)[0] takes the first element of the result set: <pmcid id="3315"></pmcid>
            # 2) soup(k)[0].get('id') reads the value of the property 'id': 3315
            v = soup(k)[0].get('id')
            d[k] = v
            selected_id = v
            if k == keys[0] or k == keys[1]:
                # prefer pmcid or uixrid, that are simple integers. uid starts with 'CXR'
                # example: pmcid=3700, uixrid=3700, uid=CXR3700
                # break as soon as you find one of the first two keys
                break
    assert selected_id  # is not None and is not empty, fail otherwise
    return {"id": selected_id}

def parse_medical_texts(soup):
    a = soup.abstract
    ats = a.find_all('abstracttext')
    res = {}
    valid_labels = ["impression", "indication", "findings", "comparison"]
    for at in ats:
        label = at.get('label').lower()
        if label in valid_labels:
            res[label] = at.text
    return res

def parse_mesh_terms(soup):
    mt = soup.mesh
    res = {}
    if mt:
        mt_major = mt.find_all('major')
        mt_minor = mt.find_all('minor')
        if mt_major:
            res["orig_mesh_terms"] = [major.text for major in mt_major if major.text]
        if mt_minor:
            res["minor_mesh"] = [minor.text for minor in mt_minor if minor.text]
    return res

def parse_automatic_terms(soup):
    mt = soup.mesh
    res = {}
    terms = []
    if mt:
        mt_auto = mt.find_all('automatic')
        if mt_auto:
            terms = [term.text for term in mt_auto if term.text]
    res["orig_auto_term"] = terms
    return res

def parse_images(soup):
    res = []
    imgs = soup.find_all('parentimage')
    for img in imgs:
        d = {}
        if img.caption:
            d["image_caption"] = img.caption.text
        if img.url:
            p = img.url.text  # this is an absolute path
            fn = filename_from_path(p, keep_extension=False)
            # dataset contains png images, but paths in reports point to (old) jpeg versions
            d["image_filename"] = fn + '.png'
        else:
            print('FATAL: NO img.url')
            exit()
        res.append(d)
    return res  # {"images": res}


def parse_single_report(filepath, verbose=False):
    with open(filepath, "r", encoding="utf-8") as fin:
        xml = fin.read()
    soup = BeautifulSoup(xml, "lxml")
    parsed = {}
    parsed.update(parse_id(soup))
    parsed.update(parse_medical_texts(soup))
    parsed.update(parse_mesh_terms(soup))
    parsed.update(parse_automatic_terms(soup))
    images = parse_images(soup)
    parsed["image_filename"] = [d["image_filename"] for d in images]
    parsed["filename"] = os.path.basename(filepath)
    return parsed

def parse_reports(txt_fld, ext="xml", verbose=False, dev=False):
    reports = []
    for i, fn in enumerate(tqdm( [ join(txt_fld, fn) for fn in os.listdir(txt_fld) if (ext is None or fn.endswith(ext)) ])):
        reports.append(parse_single_report(fn))
    return reports


def main(ds_home, out_fld):
    reports_fld = join(ds_home, "text")
    images_fld = join(ds_home, "image")
    reports = parse_reports(reports_fld)
    reports = pd.DataFrame.from_records(reports).set_index("id")
    reports.sort_index(inplace=True)

    reports["n_images"] = reports["image_filename"].apply(lambda l: len(l))
    reports["n_orig_mesh_terms"] = reports["orig_mesh_terms"].apply(lambda l: len(l))
    reports["n_orig_auto_terms"] = reports["orig_auto_term"].apply(lambda l: len(l))

    out_fn = join(out_fld, "reports_raw.pkl")
    reports.to_pickle( out_fn )
    print(f"saved {out_fn}, all done.")



if __name__ == "__main__":
    fire.Fire(main)