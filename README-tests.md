To run performance tests please use bin/perftest_8gpu.sh and bin/perftest_cpu.sh files as examples.
For example:
```
BATCH_SIZE=128 SEQ_LEN=100 NUM_ITERS=100 NUM_RNN_LAYERS=5 HIDDEN_SIZE=3072 bin/perftest_8gpu.sh

BATCH_SIZE=128 SEQ_LEN=100 NUM_ITERS=100 NUM_RNN_LAYERS=5 HIDDEN_SIZE=3072 bin/perftest_cpu.sh
```

The result is in 'SESSION time' as hours:minutes:seconds spent in TF.graph evaluation.

