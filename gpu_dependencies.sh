#!/bin/sh

set -e

sudo apt-get -y update
sudo apt-get -y install build-essential gcc-multilib dkms
# sudo apt-get -y install -y install g++ gfortran
# sudo apt-get -y build-dep libopenblas-dev
# sudo apt-get -y install liblapack-dev
# sudo apt-get -y install libavbin-dev
# sudo apt-get -y build-dep nvidia-cuda-toolkit

# some day ubuntu will release a working CUDA repo
# until that day comes, we need to get it straight from nvidia
if [ ! -e 'cuda_7.5.18_linux.run' ]; then
		wget 'http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run'
fi
chmod +x cuda_7.5.18_linux.run
sudo service lightdm stop
./cuda_7.5.18_linux.run --extract=$HOME
sudo ./cuda-linux64-rel-7.5.18-19867135.run
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig

echo "Dependencies for GPU training successfully installed."
echo "You may have to reboot before GPU acceleration will work."
