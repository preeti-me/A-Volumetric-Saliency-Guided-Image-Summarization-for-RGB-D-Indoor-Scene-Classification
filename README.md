# A Volumetric-Saliency-Guided-Image-Summarization-for-RGB-D-Indoor-Scene-Classification
This repo contains the data used in the paper, " A Volumetric Saliency Guided Image Summarization for RGB-D Indoor Scene Classification", IEEE Transactions on Circuits and Systems for Video Technology, 2024.

# Project page
[volumetricsaliencybasedsummary](https://sites.google.com/iitj.ac.in/volumetricsaliencybasedsummary?usp=sharing)


# Dataset structure
The graph dataset is structured as follows:

```bash
buildingdata
└── building1
    ├── name               # building name
    ├── rooms         
    │   ├── name               # rooms name ex: kitchen, bedroom
    │   ├── rgbimages          # color images
    │   ├── depthimages        # depth images
    │   ├── salientobjects     # salient objects properties
    │       ├── labels     
    │       ├── Volume
    │       ├── centroid
    │       ├── corners    
    │       ├── orientation  
    │       └── objectPointClouds 
    │   ├── adjacency matrix
    │   └── saliency graph     # graph consists of salient objects as nodes and adjacency matrix
└── ...

