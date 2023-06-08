# SBT: Single Branch Transformer Tracking

### This the reproduced version of our work "Correlation-Aware Deep Tracking".  You can find it in https://arxiv.org/abs/2203.01666 



## Abstract
Robustness and discrimination power are two fundamental requirements in visual object tracking. In most tracking paradigms, we find that the features extracted by the popular Siamese-like networks cannot fully discriminatively model the tracked targets and distractor objects, hindering them from simultaneously meeting these two requirements. While most methods focus on designing robust correlation operations, we propose a novel target-dependent feature network inspired by the self-/cross-attention scheme. In contrast to the Siamese-like feature extraction, our network deeply embeds cross-image feature correlation in multiple layers of the feature network. By extensively matching the features of the two images through multiple layers, it is able to suppress non-target features, resulting in instance-varying feature extraction. The output features of the search image can be directly used for predicting target locations without extra correlation step. Moreover, our model can be flexibly pre-trained on abundant unpaired images, leading to notably faster convergence than the existing methods. Extensive experiments show our method achieves the state-of-the-art results while running at real-time. Our feature networks also can be applied to existing tracking pipelines seamlessly to raise the tracking performance.

## Results
We obtain the state-of-the-art results on several benchmarks while running at high speed. 
More results are coming soon. 

<table>
  <tr>
    <th>Model</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>GOT-10k<br>SR0.5 (%)</th>
    <th>GOT-10k<br>SR0.75 (%)</th>
    <th>Speed<br></th>
    <th>Params<br></th>
  </tr>
  <tr>
    <td>SBT-base</td>
    <td>69.7</td>
    <td>79.9</td>
    <td>64.1</td>
    <td>40fps</td>
    <td>25.1M</td>
  </tr>
  <tr>
    
<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>Suc.(%)</th>
    <th>LaSOT<br>Pre.</th>
    <th>LaSOT<br>Norm. Pre.</th>
    <th>Speed<br></th>
    <th>Params<br></th>
  </tr>
  <tr>
    <td>SuperSBT-base</td>
    <td>68.0</td>
    <td>73.9</td>
    <td>77.8</td>
    <td>40fps</td>
    <td>25.1M</td>
  </tr>
  <tr>
    
#### Install dependencies
* Docker image
    ```
    We also provide a docker image for reproducing our results:
    jaffe03/dualtfrpp:latest
    ```   
* Create and activate a conda environment 
    ```bash
    conda create -n SBT python=3.7
    conda activate SBT
    ```  
* Install PyTorch
    ```bash
    conda install -c pytorch pytorch=1.6 torchvision=0.7.1 cudatoolkit=10.2
    ```  

* Install other packages
    ```bash
    conda install matplotlib pandas tqdm
    pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    sudo apt-get install libturbojpeg
    pip install pycocotools jpeg4py
    pip install wget yacs
    pip install shapely==1.6.4.post2
    pip install mmcv timm
    ```  
* Setup the environment                                                                                                 
Create the default environment setting files.

## Acknowledgement
This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking) based on **Pytorch**, 
also borrowing from [PySOT](https://github.com/STVIR/pysot), [GOT-10k](https://github.com/got-10k/toolkit) and [Vision Transformer](https://github.com/lucidrains/vit-pytorch), such as [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [PVT](https://github.com/whai362/PVT), [Twins](https://github.com/Meituan-AutoML/Twins). 
We would like to thank their authors for providing great code and framework. 

## Contacts
* Fei Xie, Shanghai Jiao Tong University, China, 372998044@qq.com
      
    
## Citing SBT
If you find SBT useful in your research, please consider citing:
```bibtex
@inproceedings{
Xie2022sbt, 
title={Correlation-Aware Deep Tracking},
author={Fei Xie, Chunyu Wang, Guangting Wang, Yue Cao, Wankou Yang, Wenjun Zeng},
booktitle={Conference on Computer Vision and Pattern Recognition},
year={2022},
}
```
