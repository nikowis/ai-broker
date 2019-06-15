# -*- coding: utf-8 -*-
"""broker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12YNxxE7-OSc-5FxiNt85Cw_PRi13-rkg
"""

from google.colab import drive
drive.mount('/content/drive/')


!apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

!pip install h5py

!ls ./drive/My\ Drive/ai-broker/scripts
!python3 ./drive/My\ Drive/ai-broker/scripts/nn_benchmark.py