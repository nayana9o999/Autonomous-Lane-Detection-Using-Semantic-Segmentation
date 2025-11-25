# Autonomous-Lane-Detection-Using-Semantic-Segmentation
Semantic segmentation-based lane detection using DeepLabV3+
This repository contains the initial setup for a project aimed at detecting road lane markings using DeepLabV3+ semantic segmentation. The system is designed for Indian road conditions, where lane markings are often inconsistent, occluded, or partially visible.

Right now, the project includes:

1.Dataset loading code

2.Preprocessing and normalization

3.Mask alignment checks

4.Base DeepLabV3+ model setup

Training and inference are still in progress, and the final output (segmented lanes + overlay on video) will be added after model training is completed.

The goal of the project is to:

Train DeepLabV3+ on the Semantic Segmentation Dataset of Indian Roads

1.Predict lane masks from input images

2.Post-process the masks to extract clean lane boundaries

3.(Optional) Fit polynomial curves to visualize smooth lane lines on real driving videos

More updates will be pushed here as the training and testing steps are completed.
