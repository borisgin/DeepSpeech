import glob
import numpy as np
from collections import defaultdict


report_dir = './REPORT/'

files = sorted(glob.glob(report_dir + '*TEST*'))

data = open(files[0], 'r').readlines()
report = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))))

for f in files:
    try:
        hw = f.split('_')[-3]
        data = open(f, 'r').readlines()
        params = data[2].split()
        seq_len = int(params[0].split('=')[-1])
        num_iters = int(params[1].split('=')[-1])
        num_rnn = int(params[12])
        hs = int(params[19])
        bs = int(params[23])
        oom = np.any(['OOM' in line for line in data])
        if not oom:
            line = data[-1]
            t = line.split(',')[0].split()[-1]
            h, m, s = t.split(':')
            t_s = (int(h)*60 + int(m))*60 + float(s)
            report[num_rnn][hs][seq_len][num_iters][bs][hw] = t_s
        else:
            report[num_rnn][hs][seq_len][num_iters][bs][hw] = 'OOM'
    except:
        pass

csv = open(report_dir + 'report.csv', 'w')
csv.write('# GRU layers,Hidden layer size,# timesteps (seq len),# iters,Batch size,Dataset size(seq),GPU time (s), GPU seq/s,GPU real-time ratio,CPU time (s),CPU seq/s,CPU real-time ratio,GPU/CPU ratio\n')
for num_rnn in sorted(report.keys()):
    for hs in sorted(report[num_rnn].keys()):
        for sl in sorted(report[num_rnn][hs].keys()):
            for num_iters in sorted(report[num_rnn][hs][sl].keys()):
                for bs in sorted(report[num_rnn][hs][sl][num_iters]):
                    csv_str = '{},{},{},{},{},{},'.format(num_rnn,
                                                    hs,
                                                    sl,
                                                    num_iters,
                                                    bs,
                                                    num_iters * bs)
                    for hw in sorted(report[num_rnn][hs][sl][num_iters][bs]):
                        t_s = report[num_rnn][hs][sl][num_iters][bs][hw]
                        if t_s != 'OOM':
                            seq_per_sec = (bs*num_iters)/t_s
                            csv_str += '{:.3f},{:.3f},{:.3f},'.format(t_s, 
                                                                      seq_per_sec, 
                                                                      seq_per_sec * sl / 100)
                        else:
                            csv_str += 'OOM,OOM,OOM,'
                    t_s_gpu = report[num_rnn][hs][sl][num_iters][bs]['1GPU']
                    if t_s_gpu != 'OOM':
                        csv_str += '{:.3f}'.format(report[num_rnn][hs][sl][num_iters][bs]['CPU']/t_s_gpu)
                    else:
                        csv_str += 'OOM'
                    csv.write(csv_str+'\n')
csv.close()
