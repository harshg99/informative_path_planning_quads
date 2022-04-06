FROM nvidia/cuda:11.6-base-ubuntu20.04

RUN apt-get update \
    && apt-get install -y \
      build-essential \
      cmake \
      cppcheck \
      gdb \
      git \
      lsb-release \
      software-properties-common \
      sudo \
      neovim \
      wget \
      net-tools \
      iputils-ping \
      tmux \
      locales \
      python3-pip \
      curl \
      vim \
      ffmpeg\
    && apt-get clean

# Conda setup
RUN curl -0 https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

RUN conda env create -n torchRL python=3.8
RUN conda init
RUN conda activate torchRL
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


RUN mkdir /home/code/
RUN cd /home/code

# Set up code for training
RUN git clone https://github.com/ljarin/Active-sampling-with-motion-primitives.git

# Setup motion primitives
RUN apt-get update\
        && apt-get install -y libeigen3-dev \
            libtbb-dev\
            libgtest-dev\
            python3-vcstool\
        && apt-get clean

RUN git clone https://github.com/ljarin/motion_primitives.git
RUN cd motion_primitives_py && pip3 install -e .

# Run remaining install dependencies
RUN pip install -r requirements.txt

# ROS Setup
# RUN sudo apt-get update \
#     && sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
#     && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
#     && sudo apt-get update \
#     && sudo apt-get install -y \
#       python3-catkin-tools \
#       python3-rosdep \
#       python3-rosinstall \
#       python3-vcstool \
#       ros-noetic-catkin \
#       ros-noetic-rosbash \
#       ros-noetic-desktop \
#       ros-noetic-pcl-ros \
#       ros-noetic-tf2-geometry-msgs \
#     && sudo rosdep init \
#     && rosdep update

