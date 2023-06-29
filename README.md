# LMCNet
Code for _"Local-to-Global Mutual-view Prototype Calibration for Few-shot Image Classification"_(LMCNet).<br>

## Requirements
* scipy==1.10.1<br>
* torch==1.13.1<br>
* torchaudio==0.13.1<br>
* torchvision==0.14.1<br>

## Abstract
Existing Few-Shot Learning (FSL) methods commonly use an episode-based training strategy to generalize the knowledge from base classes to novel classes, but this inevitably has a sample bias problem that makes it difficult for the model to maintain good performance in the novel class. In the paper, we propose a Local-to-global Mutual-view prototype Calibration Network (LMCNet) to alleviate this problem in two ways. First, we utilize global information to correct support features in each episode. Second, we propose a Local-view Prototype set Generation (LPG) algorithm, which generates local-view prototype sets from support local features and creates query instance-level prototypes based on query images, thus eliminating the requirement to use the global feature of support images as class prototypes. In this way, the model can retain more discriminative features, and enhances generalization capability. Extensive experiments on multiple benchmark datasets verify the state-of-the-art effectiveness of our method and confirm its validity.
