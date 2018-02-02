#!/usr/bin/python
import numpy as np
import six
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse


def parse_log(log):
    train = []
    val = []
    with open(log, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.replace(',', '')
            line = six.u(line)
            if 'Training' in line:
                if 'WER' in line:
                    epoch, _, loss, _ = [float(w) for w in line.split() if w[0].isnumeric()]
                else:
                    epoch, loss = [float(w) for w in line.split() if w[0].isnumeric()]
                train.append( [epoch, loss] )
            elif 'Validation' in line:
                if 'WER' in line:
                    epoch, _, loss, _ = [float(w) for w in line.split() if w[0].isnumeric()]
                else:
                    epoch, loss = [float(w) for w in line.split() if w[0].isnumeric()]
                val.append( [epoch, loss] )
        f.close()

    train = np.array(train)
    val = np.array(val)
    return train, val


def plot_logs(logs):
    # parse log files
    data = []
    for log in logs:
        data.append(parse_log(log))

    # plot
    COLORS = list(colors.TABLEAU_COLORS.values())
    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(16, 8))
    legend = []
    for idx, trace in enumerate(data):
        color = COLORS[idx]
        plt.plot(trace[0][:, 0], trace[0][:, 1], '-', color=color)
        plt.plot(trace[1][:, 0], trace[1][:, 1], '.-', color=color)
        log_name = os.path.basename(logs[idx])
        legend += ['Training, {}'.format(log_name),
                   'Validation, {}'.format(log_name)]
    plt.legend(legend)
    plt.title('Deep Speech')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot train/validation losses from DeepSpeech training logs')
    parser.add_argument('logs', metavar='log_file', type=str, nargs='+', help='DeepSpeech log file')
    args = parser.parse_args()

    plot_logs(args.logs)

