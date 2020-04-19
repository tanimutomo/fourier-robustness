#!/bin/bash

weights_dir=$1

for weight_file in `ls ${weights_dir}/*.pth` ;
do
  weight=${weight_file#*/}
  echo ${weight}
  python src/cifar10.py experiment=save weight=${weight}
done