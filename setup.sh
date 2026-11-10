#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies
echo "Installing base requirements..."
pip3 install -r requirements.txt

# 2. Clone and install Salad
echo "Cloning and installing Salad..."
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad

# 3. Clone and install VGGT
echo "Cloning and installing VGGT..."
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt
pip install -e ./vggt

# 4. Install Perception Encoder
echo "Cloning and installing Perception Encoder..."
git clone https://github.com/facebookresearch/perception_models.git
pip install -e ./perception_models

# 5. Install SAM 3
echo "Cloning and installing SAM 3..."
git clone  https://github.com/MIT-SPARK/SAM3_SPARK.git sam3
pip install -e ./sam3

# 6. Install Depth Anything V3 (optional, used to get approximate metric depth of the scene)
echo "Cloning and installing Depth Anything V3..."
git clone https://github.com/MIT-SPARK/DepthAnything_SPARK.git Depth-Anything-3
pip install -e ./Depth-Anything-3

# 7. Install VGGT-Omega
echo "Cloning and installing VGGT-Omega..."
git clone https://github.com/MIT-SPARK/Omega_SPARK.git vggt-omega
pip install -e ./vggt-omega

# 8. Install current repo in editable mode
echo "Installing current repo..."
pip install -e .

# 9. asmk pulls in faiss-cpu, which shares the `faiss` package dir with faiss-gpu-cu12
# and overwrites its GPU libraries. Remove both and reinstall only the GPU build.
echo "Reinstalling faiss-gpu..."
pip uninstall -y faiss-cpu faiss-gpu-cu12
pip install --no-deps faiss-gpu-cu12==1.11.0

echo "Installation Complete"
