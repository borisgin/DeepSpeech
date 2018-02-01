import pandas as pd
import os
import argparse

KENLM_BIN = '/opt/kenlm/build/bin/'

def get_corpus(csv_files):
    '''
    Get text corpus from a list of CSV files
    '''
    SEP = '\n'
    corpus = ''
    for f in csv_files:
        df = pd.read_csv(f)
        corpus += SEP.join(df['transcript']) + SEP
    # remove the last SEP
    corpus = corpus[:-1]
    return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build N-gram LM model from CSV files')
    parser.add_argument('csv', metavar='csv', type=str, nargs='+', help='DeepSpeech CSV file')
    parser.add_argument('--n', type=int, help='n for n-grams', default=3)
    args = parser.parse_args()

    corpus = get_corpus(args.csv)

    path_prefix, _ = os.path.splitext(args.csv[0])
    corpus_name = path_prefix + '.txt'
    arpa_name = path_prefix + '.arpa'
    lm_name = path_prefix + '-lm.binary'
    trie_name = path_prefix + '-lm.trie'
    with open(corpus_name, 'w') as f:
        f.write(corpus)

    command = KENLM_BIN + 'lmplz --text {} --arpa {} --o {}'.format(
        corpus_name, arpa_name, args.n)
    print(command)
    os.system(command)

    command = KENLM_BIN + 'build_binary -s {} {}'.format(
        arpa_name, lm_name)
    print(command)
    os.system(command)

    command = 'native_client/generate_trie data/alphabet.txt {} {} {}'.format(
        lm_name, corpus_name, trie_name)
    print(command)
    os.system(command)
