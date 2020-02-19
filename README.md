##Calibrated Classification under Domain Shift using Differentiable Density Ratio Estimation

###Introduction

We develop a classification method under domain shift to provide accurate yet conservative classifiers with trustworthy uncertainties. Calibrated uncertainties are essential for robust decision-making when environment changes or shifts in safety-critical applications. Previous work on learning under domain shift typically focuses on aligning the source and target representations, which may produce over-confident predictions. Moreover, uncertainty calibration using validation data is not plausible when no supervision is available on the target domain. 

In this paper, we propose, **E2C2**, an end-to-end calibrated classification framework that directly accounts for the tradeoff between aligned representation and learning conservative classifiers for the target domain. Our framework is based on a novel application of a differentiable density ratio estimation network, which is trained with a classification network jointly using alternative training. This differentiable density ratio estimation network obtains learning signals directly from the optimization of target classification performance, which largely improves upon non-end-to-end methods. We also demonstrate that this framework can be applied to multiple existing domain adaptation approaches. We can produce more calibrated uncertainties for the target domain, as well as achieve competitive accuracy on benchmark datasets of domain shift.

###Usage

####Environment

* Python 2.7, Python 3.6
* PyTorch 1.2.0
* Pyro 1.1.0
* CUDA version 10.0

####File introduction

We provide the introduction to the folders and files in this section. 

* office/ and OfficeHome/ contains the raw images data. Data can be found and downloaded from https://github.com/jindongwang/transferlearning/tree/master/data
* aligned_data/ is for saving the aligned data generated from Deep CORAL models. models/ is for saving trained models
* model_layers.py defines the foundations of IW and E2C2 models, and all other Python files can be directly run
* Some programs support self-defined arguments, you can change it via command or modify the code

