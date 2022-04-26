#! /bin/bash
echo "Running code"

# # MNIST
# config_dir="./configs/vp/mnist_config.py"
# workdir="./results/mnist"
# mode="train"
#
# python main.py --config=$config_dir --workdir=$workdir --mode=$mode


# MNIST
config_dir="./configs/vp/fashion_mnist_config.py"
workdir="./results/fashion"
mode="train"

python main.py --config=$config_dir --workdir=$workdir --mode=$mode



echo "Done"