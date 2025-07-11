


# BLS-GAN
[AAAI 25] BLS-GAN: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs


## Paper
https://arxiv.org/pdf/2409.07304

## Abstract
Conventional radiography is the widely used imaging technology in diagnosing, monitoring, and prognosticating musculoskeletal (MSK) diseases because of its easy availability, versatility, and cost-effectiveness. Bone overlaps are prevalent in conventional radiographs, and can impede the accurate assessment of bone characteristics by radiologists or algorithms, posing significant challenges to conventional clinical diagnosis and computer-aided diagnosis. This work initiated the study of a challenging scenario - bone layer separation in conventional radiographs, in which separate overlapped bone regions enable the independent assessment of the bone characteristics of each bone layer and lay the groundwork for MSK disease diagnosis and its automation. This work proposed a Bone Layer Separation GAN (BLS-GAN) framework that can produce high-quality bone layer images with reasonable bone characteristics and texture. This framework introduced a reconstructor based on conventional radiography imaging principles, which achieved efficient reconstruction and mitigates the recurrent calculations and training instability issues caused by soft tissue in the overlapped regions.
Additionally, pre-training with synthetic images was implemented to enhance the stability of both the training process and the results. The generated images passed the visual Turing test, and improved performance in downstream tasks. This work affirms the feasibility of extracting bone layer images from conventional radiographs, which holds promise for leveraging layer separation technology to facilitate more comprehensive analytical research in MSK diagnosis, monitoring, and prognosis.

## Training Steps
1. **Environment Setup**
   - Install the required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure the environment supports .

2. **Dataset**
   - The dataset is on `/Data`.

3. **Model Training**
   - Start the training process with in `/train/pre_pretrain.ipynb` for pre training in Synthetic Images.
   - Start the training process with in `/train/train_pretrain.ipynb` for pre training in Synthetic Images and Real Images.

## Inference Steps
1. **Model Setup**
   - Download the pre-trained model from https://drive.google.com/drive/folders/1bK1IMLDgmJrx0TU_YhO9qwZwC2P1v8xe?usp=drive_link
   - Place the model file in the designated folder `/parameters/main_train`.

2. **Run Inference**
   - Perform inference on test data in `/predict/fake_image.ipynb` for fake overlap (Synthetic Images) images.
   - Perform inference on test data in `/predict/real_image.ipynb` for real overlap images.
   - Output results will be saved in `/results`.

## Citation
If you find this code useful, please consider citing our paper:
```
@inproceedings{
wang2024blsgan,
title={{BLS}-{GAN}: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs},
author={Haolin Wang, Yafei Ou, Prasoon Ambalathankandy, Gen Ota, Pengyu Dai, Masayuki Ikebe, Kenji Suzuki, Tamotsu Kamishima},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2024},
url={https://openreview.net/forum?id=9qijUhKEW5}
}
```
