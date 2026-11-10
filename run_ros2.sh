#!/bin/bash
# Launcher for VGGT-SLAM ROS 2 node.
# Puts conda env site-packages ahead of ROS Kilted's Python 3.12 packages
# so the robostack-compiled (Python 3.11) rclpy and cv_bridge are used.

CONDA_ENV=/home/dominator/miniconda3/envs/vggt-slam
CONDA_SP=$CONDA_ENV/lib/python3.11/site-packages

export PYTHONPATH=$CONDA_SP:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH

exec $CONDA_ENV/bin/python -u main_ros2.py "$@"
