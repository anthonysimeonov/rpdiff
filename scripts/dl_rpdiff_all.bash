#!/bin/bash

set -euo pipefail

if [ -z $RPDIFF_SOURCE_DIR ]; then printf "Please source "rpdiff_env.sh" first\n"
else

printf "\n\nDownloading quickstart pre-trained weights\n\n"
sleep 2

WEIGHTS_TARNAME=quickstart_weights.tar.gz
wget -O $WEIGHTS_TARNAME https://www.dropbox.com/s/qqm89c9acyhupvu/quickstart_weights.tar.gz?dl=0
MODEL_WEIGHTS_DIR=$RPDIFF_SOURCE_DIR/model_weights/rpdiff
mkdir -p $MODEL_WEIGHTS_DIR
mv $WEIGHTS_TARNAME $MODEL_WEIGHTS_DIR
cd $MODEL_WEIGHTS_DIR
tar -xzf $WEIGHTS_TARNAME
rm $WEIGHTS_TARNAME 
printf "Model weights copied to $MODEL_WEIGHTS_DIR\n"

printf "\n\nDownloading object assets\n\n"
sleep 2

OBJ_TARNAME=objects.tar.gz
wget -O $OBJ_TARNAME https://www.dropbox.com/s/jym0n203gh7a81v/objects.tar.gz?dl=0
OBJ_DESC_DIR=$RPDIFF_SOURCE_DIR/descriptions
mkdir -p $OBJ_DESC_DIR
mv $OBJ_TARNAME $OBJ_DESC_DIR
cd $OBJ_DESC_DIR
tar -xzf $OBJ_TARNAME
rm $OBJ_TARNAME 
printf "Object models copied to $OBJ_DESC_DIR\n"

printf "\n\nDownloading pre-collected training data\n\n"
sleep 2

BOOK_BOOKSHELF_TARNAME=book_on_bookshelf_double_view_rnd_ori.tar.gz
MUG_RACK_TARNAME=mug_on_rack_multi_large_proc_gen_demos.tar.gz
CAN_CABINET_TARNAME=can_in_cabinet_stack.tar.gz
wget -O $BOOK_BOOKSHELF_TARNAME https://www.dropbox.com/s/crnlfucbogwkc3i/book_on_bookshelf_double_view_rnd_ori.tar.gz?dl=0
wget -O $MUG_RACK_TARNAME https://www.dropbox.com/s/op1jft08hz5pelw/mug_on_rack_multi_large_proc_gen_demos.tar.gz?dl=0 
wget -O $CAN_CABINET_TARNAME https://www.dropbox.com/s/eof4ahcjx94wgua/can_in_cabinet_stack.tar.gz?dl=0

TRAIN_DATA_DIR=$RPDIFF_SOURCE_DIR/data/task_demos
mkdir -p $TRAIN_DATA_DIR

mv $BOOK_BOOKSHELF_TARNAME $TRAIN_DATA_DIR
mv $MUG_RACK_TARNAME $TRAIN_DATA_DIR
mv $CAN_CABINET_TARNAME $TRAIN_DATA_DIR

cd $TRAIN_DATA_DIR

tar -xzf $BOOK_BOOKSHELF_TARNAME
tar -xzf $MUG_RACK_TARNAME
tar -xzf $CAN_CABINET_TARNAME

rm $BOOK_BOOKSHELF_TARNAME
rm $MUG_RACK_TARNAME
rm $CAN_CABINET_TARNAME

printf "Training data copied to $TRAIN_DATA_DIR\n"
fi
