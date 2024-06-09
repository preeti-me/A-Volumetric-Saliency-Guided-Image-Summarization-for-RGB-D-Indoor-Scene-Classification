# Image-Summarization
This repo contains the implementation of the paper, "A Volumetric Saliency Guided Image Summarization for RGB-D Indoor Scene Classification", IEEE Transactions on Circuits and Systems for Video Technology, 2024.


# Project page
[volumetricsaliencybasedsummary](https://sites.google.com/iitj.ac.in/volumetricsaliencybasedsummary?usp=sharing)

# Dataset
 You can download the NYU depth v2 dataset from https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

# Code
seg_as_newimg.m : extract each segment as a new image by masking and zero padding the original RGB image.

volumetric_saliency.m : Volumetric saliency map prediction.

Visual_summ.m : Generate saliency-guided image summary.

# Citation
If you use volumetric saliency in your research, please use the following BibTeX entry.

```bash
@article{meena2024volumetric,
  title={A Volumetric Saliency Guided Image Summarization for RGB-D Indoor Scene Classification},
  author={Meena, Preeti and Kumar, Himanshu and Yadav, Sandeep},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE},
  DOI= 10.1109/TCSVT.2024.3412949
}
