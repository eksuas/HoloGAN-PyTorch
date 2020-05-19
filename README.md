# HoloGAN-PyTorch-
HoloGAN implementation in PyTorch

## Building the development environment:
Install Anaconda 3.7 from the website: https://www.anaconda.com/products/individual

Check cuda version if the Cuda is avaliable in the system so that we can work on GPU.
```markdown  
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:04_Central_Daylight_Time_2018
Cuda compilation tools, release 10.0, V10.0.130
```
Check the website for proper installation command: https://pytorch.org/get-started/locally/. Below command is for a stable version of PyTorch on Linux. The conda installation is recommanded. According to the cuda version change the cudatoolkit version.


Create a new conda environment for HoloGAN:
```markdown  
$ conda create -n hologan python=3.7
```

Activate the hologan environment:
```markdown  
$ conda activate hologan
```

Install required libraries
```markdown  
$ conda install pytorch torchvision cpuonly -c pytorch
$ pip install scipy==1.1.0
```

to run:
python main.py --batch-size 1 --max-epochs 1 --rotate-azimuth --image-path ../dataset/fake/celebA/

ro run other:
python main.py ./config_HoloGAN.json --dataset celebA --crop input_height 108 --output_height 64 --batch-size 1
