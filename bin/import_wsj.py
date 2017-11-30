#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import fnmatch
import pandas
import subprocess
import unicodedata
import wave
import glob
import codecs


import re

def preprocess_text(label):
    # For now we can only handle [a-z ']
    '''
    marks = ["\,COMMA",
             "\.PERIOD",
             '\\"DOUBLE\-QUOTE',
             "\-HYPHEN",
             "\.POINT",
             "\%PERCENT",
             "\--DASH",
             "\&AMPERSAND",
             "\:COLON",
             "\)RIGHT\-PAREN",
             "\(LEFT\-PAREN",
             "\;SEMI\-COLON",
             "\?QUESTION\-MARK",
             "\\'SINGLE\-QUOTE",
             "...ELLIPSIS",
             "/SLASH",
             "}RIGHT-BRACE",
             "{LEFT-BRACE",
             "!EXCLAMATION-POINT",
             "+PLUS",
             "=EQUALS",
             "#SHARP-SIGN",
             "-MINUS"
    ]
    '''

    p = re.compile(' \(\S*\)|\[\S*\]')
    label = p.sub('', label)

    label = label.replace('\\"DOUBLE\-QUOTE', "DOUBLE QUOTE")
    label = label.replace('\\"DOUBLE-QUOTE', "DOUBLE QUOTE")
    label = label.replace("\\'SINGLE\-QUOTE", "SINGLE QUOTE")
    label = label.replace("\)RIGHT\-PAREN", "RIGHT PAREN")
    label = label.replace("\(LEFT\-PAREN", "LEFT PAREN")
    label = label.replace("\;SEMI\-COLON", "SEMI COLON")
    label = label.replace("\?QUESTION\-MARK", "QUESTION MARK")
    label = label.replace("}RIGHT-BRACE", "RIGHT BRACE")
    label = label.replace("{LEFT-BRACE", "LEFT BRACE")
    label = label.replace("!EXCLAMATION-POINT", "EXCLAMATION POINT")
    label = label.replace("#SHARP-SIGN", "SHARP SIGN")

    label = label.replace("\-", " ")
    label = label.replace("Ms\.", "MISS")
    label = label.replace("Mr\.", "MISTER")

    marks = [",", ".", "-", "%", "&", ":",
             ")", "(", ";", "?", "...", "/",
             "}", "{", "!", "+", "=", "#",
             "~", "*"]

    for m in marks:
        label = label.replace(m, "")

    label = label.replace("\\", "")

    label = label.replace("\\", "")
    label = label.replace("\"", "'")
    label = label.replace("\'", "'")
    label = label.replace("_", "")

    while "  " in label:
        label = label.replace("  ", " ")

    label = label.strip().lower()

    if re.search(r"[^a-z ']", label) != None or len(label) == 0:
        return None

    return label



def _preprocess_datasets(data_dir):

    dataset_dir = os.path.join(data_dir, "CSR-I(WSJ0)", "csr_1")
    files = _download_and_preprocess_data(dataset_dir, "wsj0")

    dataset_dir = os.path.join(data_dir, "CSR-II(WSJ1)", "csr_2")
    files += _download_and_preprocess_data(dataset_dir, "wsj1")

    df_files = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

    train_files, dev_files, test_files = _split_sets(df_files)

    # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(data_dir, "wsj-train.csv"), index=False)
    dev_files.to_csv(os.path.join(data_dir, "wsj-dev.csv"), index=False)
    test_files.to_csv(os.path.join(data_dir, "wsj-test.csv"), index=False)


def _download_and_preprocess_data(dataset_dir, dataset_name):

    # convert all *.wv* files to *.wav
    target_dir = os.path.join(dataset_dir, "wav")

    wv_files = glob.glob(os.path.join(dataset_dir, "*", dataset_name, "*_s", "*", "*.wv*"))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

        for wv_file in wv_files:
            wav_filename = os.path.splitext(os.path.basename(wv_file))[0] + ".wav"
            wav_file = os.path.join(target_dir, wav_filename)
            print("converting {} to {}".format(wv_file, wav_file))
            subprocess.check_call(["sph2pipe", "-p", "-f", "rif", wv_file, wav_file])

    files = []
    # parse transcripts
    tr_dir = os.path.join(dataset_dir, 'MERGED/' + dataset_name + '/transcrp/dots/si_tr_s/')
    tr_files = glob.glob(tr_dir + '/*/*.dot')
    for tr_file in tr_files:
        lines = open(tr_file, 'r').readlines()
        for line in lines:
            transcript = preprocess_text(line)
            if transcript is None or len(transcript)==0:
                continue
            wav_file = os.path.join(target_dir, line.split('(')[-1].split(')')[0] + '.wav')
            wav_size = os.path.getsize(wav_file)
            files.append((wav_file, wav_size, transcript))

    return files





def _split_sets(filelist):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return (filelist[train_beg:train_end], filelist[dev_beg:dev_end], filelist[test_beg:test_end])


if __name__ == "__main__":
    _preprocess_datasets(sys.argv[1])
