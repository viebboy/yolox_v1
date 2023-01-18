# cd to your preferred directory and clone this repo
git clone https://github.com/KaiyangZhou/deep-person-reid.git ./data/torchreid
#
# # create environment
# cd deep-person-reid/
# conda create --name torchreid python=3.7
# conda activate torchreid
#
# # install dependencies
# # make sure `which python` and `which pip` point to the correct path
wget https://github.com/pietrodn/grpcio-mac-arm-build/releases/download/1.51.1/grpcio-1.51.1-cp38-cp38-macosx_11_0_arm64.whl -P ./data/
pip install grpcio-1.51.1-cp38-cp38-macosx_11_0_arm64.whl
pip install -r ./data/torchreid/requirements.txt
#
# # install torch and torchvision (select the proper cuda version to suit your machine)
# conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
#
# # install torchreid (don't need to re-build it if you modify the source code)
# python setup.py develop
pip install -e ./data/torchreid/
pip install -r ./requirements.txt
