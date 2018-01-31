import pandas as pd
import os
import argparse

KENLM_BIN = '/opt/kenlm/build/bin/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build N-gram LM model from CSV files')
    parser.add_argument('csv', metavar='csv', type=str, nargs='+', help='DeepSpeech CSV file')
    parser.add_argument('--N', type=int, help='N for N-grams', default=3)
    args = parser.parse_args()


    SEP = '\n'
    corpus = ''
    for f in args.csv:
        df = pd.read_csv(f)
        corpus += SEP.join(df['transcript']) + SEP
    # remove the last SEP
    corpus = corpus[:-1]

    path_prefix, _ = os.path.splitext(args.csv[0])
    corpus_name = path_prefix + '.txt'
    arpa_name = path_prefix + '.arpa'
    lm_name = path_prefix + '-lm.binary'
    trie_name = path_prefix + '-lm.trie'
    with open(corpus_name, 'w') as f:
        f.write(corpus)


    os.system(KENLM_BIN + 'lmplz --text {} --arpa {} --o {}'.format(
        corpus_name, arpa_name, args.N))

    os.system(KENLM_BIN + 'build_binary -s {} {}'.format(
        arpa_name, lm_name))

    os.system('native_client/generate_trie data/alphabet.txt {} {} {}'.format(
        lm_name, corpus_name, trie_name))