
# for KAUST cluster
module load cuda/11.7.0
module load gcc/7.5.0
module load eigen

# for aws ubuntu.  install eigen
#sudo apt update && sudo apt upgrade
#sudo apt install libeigen3-dev

 # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"

# use python venv
python -m venv venv_magic123
source venv_magic123/bin/activate

# use conda
# conda create -n magic123 python=3.10 ipython -y
# conda activate magic123

pip3 install torch torchvision
pip3 install -r requirements.txt
bash scripts/install_ext.sh