export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

git clone --recursive https://github.com/FurkanGozukara/Trellis_2_3D_Generator

cd Trellis_2_3D_Generator

git reset --hard

git pull

git clone https://github.com/FurkanGozukara/UniRig

cd UniRig

git reset --hard

git pull

cd ..

# Remove existing Python 3.11 RC version
echo "Removing existing Python 3.11 (RC)..."
sudo apt-get remove -y python3.11 python3.11-venv python3.11-dev python3.11-distutils 2>/dev/null
sudo apt-get autoremove -y

# Add deadsnakes PPA
echo "Adding deadsnakes PPA..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Pin deadsnakes packages higher than the Ubuntu repo to prevent RC installs
sudo tee /etc/apt/preferences.d/deadsnakes-python3.11 > /dev/null <<EOF
Package: python3.11 python3.11-*
Pin: release o=LP-PPA-deadsnakes-ppa
Pin-Priority: 1000
EOF

# Show what will be installed
echo "Available Python 3.11 versions:"
apt-cache policy python3.11

# Install Python 3.11 from deadsnakes (candidate is already 3.11.15)
echo "Installing latest stable Python 3.11..."
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

python3.11 -m venv venv

source ./venv/bin/activate

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_trellis.txt



uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/flex_gemm-1.0.0-cp311-cp311-linux_x86_64.whl
uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/o_voxel-0.0.1-cp311-cp311-linux_x86_64.whl
uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/spconv_cu130-2.3.8-cp311-cp311-linux_x86_64.whl
uv pip install --no-deps git+https://github.com/Luo-Yihao/FaithC.git

python Trellis_2_3D_Generator/ensure_ultrashape_source.py

export HF_HUB_ENABLE_HF_TRANSFER=1

python3 HF_model_downloader.py
