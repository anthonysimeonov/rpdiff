#!/bin/bash

set -euo pipefail

if [ -z $RPDIFF_SOURCE_DIR ]; then printf "Please source "rpdiff_env.sh" first"
else
WEIGHTS_TARNAME=quickstart_weights.tar.gz
wget -O $WEIGHTS_TARNAME https://www.dropbox.com/s/qqm89c9acyhupvu/quickstart_weights.tar.gz?dl=0
MODEL_WEIGHTS_DIR=$RPDIFF_SOURCE_DIR/model_weights/rpdiff
mkdir -p $MODEL_WEIGHTS_DIR
mv $WEIGHTS_TARNAME $MODEL_WEIGHTS_DIR
cd $MODEL_WEIGHTS_DIR
tar -xzf $WEIGHTS_TARNAME
rm $WEIGHTS_TARNAME 
printf "Model weights copied to $MODEL_WEIGHTS_DIR\n"

fi
