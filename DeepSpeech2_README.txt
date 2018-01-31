=== CUDA ======================================================================


=== PRE-BUILD =================================================================

sudo apt-get install sox libsox-dev swig

==== BAZEL CLEAN ==============================================================

?install bazel 0.5.4 from https://github.com/bazelbuild/bazel/releases
rm -rf ~/.cache/bazel/

=== INSTALL TF=================================================================

git clone https://github.com/tensorflow/tensorflow/tree/r1.5
cd tensorflow
./configure
  cudnn: /usr/lib/x86_64-linux-gnu

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/*.whl

===== DeepSpeech 2 =====================================================

export LD_LIBRARY_PATH=/usr/local/lib/python2.7/dist-packages/tensorflow:$LD_LIBRARY_PATH

git clone https://github.com/borisgin/DeepSpeech/ -b dev

sudo pip install requirements.txt

cd tensorflow
ln -s ../DeepSpeech/native_client/ ./
bazel build -c opt --config=opt --config=cuda  //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //native_client:deepspeech //native_client:deepspeech_utils //native_client:libctc_decoder_with_kenlm.so //native_client:generate_trie

cd ../DeepSpeech/native_client
cp ../../tensorflow/bazel-bin/native_client/*.so .


