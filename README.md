# PP-NAS: Searching for Plug-and-Play Blocks on Convolutional Neural Networks

---

The code for [Biluo Shen, et al. PP-NAS: Searching for Plug-and-Play Blocks on Convolutional Neural Networks](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/html/Shen_PP-NAS_Searching_for_Plug-and-Play_Blocks_on_Convolutional_Neural_Network_ICCVW_2021_paper.html) in 2021 ICCV Workshop. 

<p align="center">
    <img src="img/Fig 1.jpg" alt="PP-NAS" width="80%">
</p>

## Requirements

- tensorflow == 2.3.4

- tensorflow_probability == 0.11.1

- [hanser](https://github.com/sbl1996/hanser) 

- Python >= 3.6

- TPU environment

## Search

```
""" 
    Change the dataset (CIFAR-10/100) and hyperparameters in codes.
    Weights for loss terms are defined in supernet_CIFAR.py
"""
python search.py
```

## Train

```
""" 
    Change the genotype, dataset (CIFAR-10/100) and hyperparameters in codes.
"""
python train.py
```

## Citation

If you refer to this work in your reseach, please cite our [paper](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/html/Shen_PP-NAS_Searching_for_Plug-and-Play_Blocks_on_Convolutional_Neural_Network_ICCVW_2021_paper.html):

```
@article{Shen2021PPNASSF,
  title={PP-NAS: Searching for Plug-and-Play Blocks on Convolutional Neural Network},
  author={Biluo Shen and Anqi Xiao and Jie Tian and Zhenhua Hu},
  journal={2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2021},
  pages={365-372}
}
```