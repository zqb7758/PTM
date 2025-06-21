# PTM
Implementation Code for paper "Enhancing transferability of adversarial examples by random perspective tilting."
=======

## Requirements
* python == 3.8.20
* pytorch == 1.8.0
* torchvision == 0.9.0
* numpy == 1.22.4
* pandas == 1.3.5
* scipy == 1.7.3
* pillow == 9.3.0
* pretrainedmodels == 0.7.4
* tqdm == 4.67.1


## Qucik Start
### Prepare the data and models.
1. We have prepared the ImageNet-compatible dataset in this program and put the data in **'./dataset/'**, and you should unrar the **'./dataset/images.rar'** before using.

2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-152, Res-101) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish. For all models(.npy) to be used, please download them from [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model), and make sure you put them in the right directory **'./models/'**.

3. The adversarially trained models (i.e,  adv_inc_v3， ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories.

### PTM Attack Method
The traditional baseline attacks and our proposed PTM attack methods are in the file __"Attacks.py"__.
All the provided codes generate adversarial examples on Inception_v3 model. If you want to attack other models, replace the model in **main()** function.

### Runing attack
1. You can run our proposed attack as follows. 
```
python attacks.py
```
We also provide the implementations of other baseline attack methods in our code, just change them to the corresponding attack methods in the **main()** function.

2. The generated adversarial examples would be stored in the directory **./inceptionv3_ptm_outputs**. Then run the file **verify.py** to evaluate the success rate of each model used in the paper:
```
python verify.py
```
## Acknowledgments
The codes mainly references: [STM](https://github.com/Zhijin-Ge/STM) and [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model)

## Citation
If you use this code for your research, please cite our paper.
```
@articles{,
     title={{Enhancing Transferability of Adversarial Examples by Random Perspective Tilting}},
     author={},
     booktitle={},
     year={2025}
}
```
