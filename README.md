<p align="center">
  <p align="center">
    <h1 align="center">Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis (Highlight)</h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://zikai1.github.io/"><strong>Mingyang Zhao</strong></a>
    ·
    <a target="_blank" href="https://xiaowuga.github.io/"><strong>Jingen Jiang</strong></a>
    ·
    <a target="_blank" href="https://www.ai.pku.edu.cn/info/1139/1341.htm"><strong>Lei Ma</strong></a>
    ·
    <a target="_blank" href="https://irc.cs.sdu.edu.cn/~shiqing/index.html"><strong>Shiqing Xin</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.com/citations?user=5hti_r0AAAAJ"><strong>Gaofeng Meng</strong></a>
   ·
    <a target="_blank" href="https://sites.google.com/site/yandongming/"><strong>Dong-Ming Yan</strong></a>
  </p>




<!--<font size=15> Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis (Highlight) <font size=15>-->

![](./fig/CVPR_Teaser.jpg)
This repository contains the official implementation of our CVPR 2024 paper "Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis". 

## Motivation
Non-rigid point set registration is to optimize a non-linear displacement field that accurately aligns one geometric shape with another. However, given two point sets, one acting as the source and the other as the target, non-rigid registration presents a highly ill-posed and much more complex challenge compared to the rigid counterpart. This increased complexity is primarily attributed to the additional freedom of deformations allowed in non-rigid registration, especially when dealing with shapes that exhibit large deformations. Previous approaches typically perform shape matching first and then estimate the alignment transformation based on the established correspondences via off-the-shelf registration techniques. Nevertheless, shape matching self has many outliers that may deteriorate registration. To address this problem, we explore a direct registration method for handling large deformations, without relying on shape matching.

## Implementation
- For convenience, the repository provides both **MATLAB** and **C++** implementations. 
- The MATLAB implementation is extremely **simple**, while C++ implementation is much **faster**.



## MATLAB 
```
- Step 1: Download the directory **"matlab_code"**, which contains data normalization, registration, and denormalization implementations. 
- Step 2: Start MATLAB and run **"test_demo.m"**. This will give you an immediate registration result for the test point cloud data in the directory **"data"**. 
```
  


## C++
### Prerequisite
+ **Eigen** (1.66 or later): https://sourceforge.net/projects/boost/files/boost-binaries/
+ **libgil** (2.5.0 or later): https://libigl.github.io/
+ **MOSEK 8**: https://www.mosek.com/downloads/, MOSEK license is required.
+ **PolyCut**: http://www.cs.ubc.ca/labs/imager/tr/2018/HexDemo/, download and extract it to a folder, be aware that it has one-month license.
+ **HexEx**: https://www.graphics.rwth-aachen.de/media/resource_files/HexEx_Windows_1_01.zip, download and extract it to a folder.


## Citation
Please consider citing our work if you find it useful:

```bibtex
@inproceedings{zhao2024clustereg,
  title={Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis},
  author={Mingyang Zhao, Jingen Jiang, Lei Ma, Shiqing Xin, Gaofeng Meng, Dong-Ming Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```



## Acknowledgements
This work is partially funded by the Strategic Priority Research Program of the Chinese
Academy of Sciences (XDB0640000), National Science and Technology Major Project (2022ZD0116305), National Natural Science Foundation of China
(62172415,62272277,62376267), and the innoHK project.



