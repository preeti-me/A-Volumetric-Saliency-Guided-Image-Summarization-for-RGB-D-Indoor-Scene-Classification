# Image-Summarization
This repo contains the implementation of the paper, "A Volumetric Saliency Guided Image Summarization for RGB-D Indoor Scene Classification", IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[volumetricsaliencybasedsummary](https://sites.google.com/iitj.ac.in/volumetricsaliencybasedsummary?usp=sharing)

# Dataset
 You can download the NYU depth v2 dataset from https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
 
 GT folder contains the grountruth volumetric saliency maps for evaluation.

# Code
seg_as_newimg.m : extract each segment as a new image by masking and zero padding the original RGB image.\
volumetric_saliency.m : Volumetric saliency map prediction.\
Visual_summ.m : Generate saliency-guided image summary.

download the segmention result/label matrix from https://drive.google.com/file/d/1bpWANX0AHfu-NUP7dPlvEU5cYE8F9ytE/view?usp=sharing and put it in data folder.

# Citation
If you use volumetric saliency in your research, please use the following BibTeX entry.

```bash
@article{meena2024volumetric,
  title={A Volumetric Saliency Guided Image Summarization for RGB-D Indoor Scene Classification},
  author={Meena, Preeti and Kumar, Himanshu and Yadav, Sandeep},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}

