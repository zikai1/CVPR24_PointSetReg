# CVPR24_PointSetReg (Highlight)
<table>
    <tr>
    <td ><center><img src="./fig/CVPR_Teaser.jpg"> </center></td>
    </tr>
</table>
This repository contains the official implementation of our CVPR 2024 paper "Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis". 

- For convenience, the repository provides both **MATLAB** and **C++** implementations. 
- The MATLAB implementation is extremely **simple**, while C++ implementation is much **faster**.



## MATLAB 
- Step 1: Download the directory **"matlab_code"**, which contains data normalization, registration, and denormalization implementations. 
- Step 2: Start MATLAB and run **"test_demo.m"**. This will give you an immediate registration result for the test point cloud data in the directory **"data"**. 

  


## C++
We are re-implementing the MATLAB code using C++ for better efficiency. We will release it as soon as possible!


## Citation
If you find our work useful, please cite our paper:

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



