# EfficientDet post processing

This tool shall create videos / images that contain all objects detected by standard EfficientDet.

## How to run
Following commands will run script on CPUs and hence it will be quite slow.

    pip install -r requirements
    python3 detect.py

If you want to use GPU on your computer you can either install all necessary requirements for TensorFlow as described here:

https://www.tensorflow.org/install/pip

However it is easier to run TensorFlow Docker images that ship all necessary libs and prerequisites. To run those images on Nvidia GPUs, Docker needs to be configured to use Nvidia's container runtime. Use instructions here:
    
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker 