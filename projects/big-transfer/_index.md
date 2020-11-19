---
title: Big Transfer
---

# Big Transfer 



[![Big Transfer](https://img.youtube.com/vi/k1GOF2jmX7c/0.jpg#center)](https://www.youtube.com/watch?v=k1GOF2jmX7c)

In this project you will look at a general representation learning method called [Big Transfer](https://arxiv.org/pdf/1912.11370.pdf) that was found that is particularly good in specialized small label downstream tasks. 

> (from the abstract) Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural
networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale
up pre-training, and propose a simple recipe where by combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets.


## Tasks / Rubric (100 points)

1. Write a 2 page (excluding pictures) summary of what transfer learning (10 points)
2. Write a 2 page (excluding pictures) summary of what architectural improvements, if any, the authors have assumed on top the ResNet-xyz V1  architecture and why (20 points)
3. Write a 4 page (excluding figures) summary of the key points that need to be made so a non-expert can understand the differences between GN, WS and BN. The top 3 comparisons will make it to my class notes with attribution. (30 points)
4. Write a 2 page (excluding figures) summary of MixUp regularization and how this may help (30 points). 
5. Explain all models that are used in producing performance results (eg RetinaNet) (10 points)


NOTE:

1. 5 points will be **subtracted** if you dont issue a pull request of your work to the github with a branch name that equals your UCID. 
2. After your submission we may call you and ask you to explain what you wrote. Copying text and images for documenting your arguments are OK but make sure you understand it. The corresponding points will be subtracted up to 100% if we find that don't understand what you put in the report. 

## Extra Points (20 points)

Apply the Big Transfer technique from [this repo](https://github.com/google-research/big_transfer) in CIFAR10 downstream dataset for 1-10 examples per class (low-data regime). Match the behavior of Fig 6 for one upstream dataset of your choice and ResNet50-x3 architecture (20 points). 

## Execution Environment

Below is the script that Shawn Cicoria has tried in a Windows environment 

```shell
# tested with with python 3.8.6 and CUDA 10.1 in GTX1080

pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy<=1.19.3

# get the model:
curl https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz
copy  .\BiT-M-R50x1-ILSVRC2012.npz BiT-M-R50x1.npz

# run the model -- adding --batch_split 8 per the docs/readmd helps with out of memory issues in GPU cards.
python -m bit_pytorch.train --name cifar10_fff --model BiT-M-R50x1 --logdir ./bit_logs --dataset cifar10 --datadir .\data  --batch_split 8

```