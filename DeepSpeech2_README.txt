=== CUDA ======================================================================


=== PRE-BUILD =================================================================

sudo apt-get install sox libsox-dev swig

==== BAZEL CLEAN ==============================================================

?install bazel 0.5.4 from https://github.com/bazelbuild/bazel/releases
rm -rf ~/.cache/bazel/

=== INSTALL TF=================================================================

sudo apt-get install python-numpy python-dev python-pip python-wheel
#sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel

#sudo apt-get install cuda-command-line-tools
sudo apt-get install libcupti-dev 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 

git clone https://github.com/tensorflow/tensorflow/ -b r1.5
cd tensorflow
./configure
  cudnn: /usr/lib/x86_64-linux-gnu

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# ignore warnings
sudo pip install /tmp/tensorflow_pkg/*.whl
# make sure that pip points to python2

===== KenLM ============================================================

sudo apt-get install -y cmake libeigen3-dev libboost-dev libboost-program-options-dev libboost-system-dev  libboost-thread-dev libboost-test-dev libbz2-dev liblzma-dev
cd /opt
git clone https://github.com/kpe/kenlm
mkdir /opt/kenlm/build
cd /opt/kenlm/build
cmake ..
make

===== DeepSpeech 2 =====================================================

export LD_LIBRARY_PATH=/usr/local/lib/python2.7/dist-packages/tensorflow:$LD_LIBRARY_PATH

git clone https://github.com/borisgin/DeepSpeech/ -b dev

sudo pip install requirements.txt
sudo pip install pyxdg resampy sox

cd tensorflow
ln -s ../DeepSpeech/native_client/ ./
bazel build -c opt --config=opt --config=cuda  //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //native_client:deepspeech //native_client:deepspeech_utils //native_client:libctc_decoder_with_kenlm.so //native_client:generate_trie

cp bazel-bin/native_client/*.so native_client/


==========================================
check 
cd ../DeepSpeech2
ldd native_client/libctc_decoder_with_kenlm.so



