FROM nvidia/cuda:11.6.0-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

#ROS-Setup
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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

ENV DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration
RUN sudo apt-get update \
    && sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
    && sudo apt-get update \
    && sudo apt-get install -y \
      python3-catkin-tools \
      python3-rosdep \
      python3-rosinstall \
      python3-vcstool \
      ros-noetic-catkin \
      ros-noetic-rosbash \
      ros-noetic-desktop \
      ros-noetic-pcl-ros \
      ros-noetic-tf2-geometry-msgs \
    && sudo rosdep init \
    && rosdep update



RUN conda init

WORKDIR /home

COPY startup.sh ./
# Run startup.sh in docker to set it up
#CMD ["bash startup.sh","bash"]

## Put this in a start.sh file
#SHELL ["source","~/.bashrc"] 
#SHELL ["conda","create","-n","torch","python=3.8"]
#SHELL ["conda","activate", "torch"]
#SHELL ["conda", "install", "pytorch","torchvision","torchaudio","cudatoolkit=11.3","-c","pytorch"] 
# You should have a conda environment in your coekr container, do the following after attaching container

#CMD ["mkdir","/home/code/"]
#CMD ["cd","/home/code"]

# Set up code for training
#CMD ["git","clone","https://github.com/ljarin/Active-sampling-with-motion-primitives.git"]
#CMD ["git","checkout","Search_vGYM"]
# Setup motion primitives
#CMD ["apt-get","update",\
#        "&&","apt-get","install","-y","libeigen3-dev", \
#            "libtbb-dev",\
#            "libgtest-dev",\
#            "python3-vcstool",\
#        "&&", "apt-get","clean"]

#CMD ["git","clone","https://github.com/ljarin/motion_primitives.git"]
#CMD ["cd","motion_primitives_py","&&","pip3"," install","-e","."]

# Run remaining install dependencies
#CMD ["pip","install","-r","requirements.txt"]

