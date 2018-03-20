#!/bin/bash
export N_ITERS=100
for n in 2 3 5 ; do
  for h in 3200 ; do
    for bs in 8 32 64 128 256 ; do
      for sl in 128 256 512 ; do
        NUM_RNN_LAYERS=$n HIDDEN_SIZE=$h BATCH_SIZE=$bs SEQ_LEN=$sl NUM_ITERS=$N_ITERS bin/perftest_1gpu.sh
        NUM_RNN_LAYERS=$n HIDDEN_SIZE=$h BATCH_SIZE=$bs SEQ_LEN=$sl NUM_ITERS=$N_ITERS bin/perftest_cpu.sh
      done
    done
  done
done
python bin/parse_report.py
