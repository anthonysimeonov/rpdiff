#!/bin/bash

set -euo pipefail

if [ -z $RPDIFF_SOURCE_DIR ]; then printf "Please source "rpdiff_env.sh" first\n"
else
OBJ_TARNAME=objects.tar.gz
wget -O $OBJ_TARNAME https://www.dropbox.com/s/jym0n203gh7a81v/objects.tar.gz?dl=0
OBJ_DESC_DIR=$RPDIFF_SOURCE_DIR/descriptions
mkdir -p $OBJ_DESC_DIR
mv $OBJ_TARNAME $OBJ_DESC_DIR
cd $OBJ_DESC_DIR
tar -xzf $OBJ_TARNAME
rm $OBJ_TARNAME 
printf "Object models copied to $OBJ_DESC_DIR\n"

fi
