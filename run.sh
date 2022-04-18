#! /bin/bash
echo "Running code"

config_dir="C:\Users\20210512\Documents\MyGithub\sgm_ood\configs\default_cifar10_config.py"
workdir="./results/mnist"
mode="train"

python main.py --config=$config_dir --workdir=$workdir --mode=$mode


echo "Done"