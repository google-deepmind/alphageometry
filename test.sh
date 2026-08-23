# !/bin/bash
virtualenv -p python3 .
source ./bin/activate

export TF_CPP_MIN_LOG_LEVEL=0 

python test.py