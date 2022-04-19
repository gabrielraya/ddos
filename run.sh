#! /bin/bash
echo "Running code"

config_dir="./configs/vp/mnist_config.py"
workdir="./results/mnist"
mode="train"

python main.py --config=$config_dir --workdir=$workdir --mode=$mode
# python main.py --config=$config_dir

echo "Done"