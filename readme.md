# Do adversarial defenses improve robustness of DNNs? #
This is a repository for the article <a href="https://lego999.github.io/ml/blog/2021/09/29/AT_robustness.html">"Do adversarial defenses improve robustness of DNNs? "</a>.
## Prerequisite ##
```
python==3.6.9
torch==1.6.0
torchvision==0.7.0
seaborn==0.11.1
numpy==1.19.5
tqdm==4.56.2
advertorch==0.2.3
pandas==1.1.5
matplotlib==3.1.1
scikit-learn==0.24.2
```
## Download pre-trained models and datasets ##
### Models ###
The vanilla and PGD defense models (<a href="https://arxiv.org/abs/1706.06083">Madry et al., 2017</a>)  come from the repository <a href="https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR">Pytorch-Adversarial-Training-CIFAR</a>.
Download: <a href="https://postechackr-my.sharepoint.com/personal/dongbinna_postech_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdongbinna%5Fpostech%5Fac%5Fkr%2FDocuments%2FResearch%2FPytorch%20Adversarial%20Training%20on%20CIFAR%2D10%2FPre%2Dtrained%2Fbasic%5Ftraining&parent=%2Fpersonal%2Fdongbinna%5Fpostech%5Fac%5Fkr%2FDocuments%2FResearch%2FPytorch%20Adversarial%20Training%20on%20CIFAR%2D10%2FPre%2Dtrained">vanilla</a>, <a href="https://postechackr-my.sharepoint.com/personal/dongbinna_postech_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdongbinna%5Fpostech%5Fac%5Fkr%2FDocuments%2FResearch%2FPytorch%20Adversarial%20Training%20on%20CIFAR%2D10%2FPre%2Dtrained%2Fpgd%5Fadversarial%5Ftraining&parent=%2Fpersonal%2Fdongbinna%5Fpostech%5Fac%5Fkr%2FDocuments%2FResearch%2FPytorch%20Adversarial%20Training%20on%20CIFAR%2D10%2FPre%2Dtrained">PGD defense</a><br/>
The DDPM data augmentation model (<a href="https://arxiv.org/abs/2103.01946">Rebuffi et al., 2021</a>) comes from their official repository <a href="https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness"> Fixing Data Augmentation to Improve Adversarial Robustness</a>
Download: <a href="https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_resnet18_ddpm.pt">DDPM</a>

### Datasets ###
Only <a href="https://arxiv.org/abs/1807.01697">CIFAR-10-C</a> needs to be downloaded manually. Download: <a href="https://zenodo.org/record/2535967#.YaJm7cfMKjg">CIFAR-10-C and CIFAR-10-P | Zenodo</a>. Other datasets in the experiments (i.e., test sets of CIFAR-10, KMNIST and SVHN) could be downloaded automatically by ```torchvision```.

### Directory structure ###
Please organize the pre-trained models and the CIFAR-10-C dataset as follows. Then the project is ready to run.
```
AD_robustness
│  .gitattributes
│  AD_robustness.ipynb
│  preact_resnet.py
│  readme.md
│  resnet.py
│  
├─AUGMENT_AT
│      ddpm.pt
│      
├─data
│  │  cifar-10-python.tar.gz $CIFAR-10 test set, need not to download manually$
│  │  test_32x32.mat $SVHN test set, need not to download manually$
│  │  
│  ├─cifar-10-batches-py $CIFAR-10 test set, need not to download manually$
│  │      ......
│  │      
│  ├─CIFAR-10-C
│  │      brightness.npy
│  │      contrast.npy
│  │      defocus_blur.npy
│  │      elastic_transform.npy
│  │      fog.npy
│  │      frost.npy
│  │      gaussian_blur.npy
│  │      gaussian_noise.npy
│  │      glass_blur.npy
│  │      impulse_noise.npy
│  │      jpeg_compression.npy
│  │      labels.npy
│  │      motion_blur.npy
│  │      pixelate.npy
│  │      saturate.npy
│  │      shot_noise.npy
│  │      snow.npy
│  │      spatter.npy
│  │      speckle_noise.npy
│  │      zoom_blur.npy
│  │      
│  └─KMNIST $KMNIST test set, need not to download manually$
│      ......
│              
└─PGD_AT
        basic_training
        pgd_adversarial_training
```
## Run ##
Run ```AD_robustness.ipynb``` and enjoy!
