# LLE-NET:A Low-Light Image Enhancement Algorithm Based on Curve Estimation
You can find more details here: https://github.com/xiujie123/LLE-NET. 
Have fun!

**The implementation of LLE-NET is for non-commercial use only.**

We also provide a Google  Drive version of our code: https://drive.google.com/drive/folders/1KLM1i06PZKDtFgkIdPgq7_fShYb2oSV6?usp=sharing.
Baidu Netdisk version : https://pan.baidu.com/s/19nLgRoxwNtYLEL_dwWJlDA (password:hnnj )


# Pytorch 
Pytorch implementation of LLE-NET

## Requirements
1. Python 3.8 
2. Pytorch 1.12.0
3. opencv
4. torchvision 0.2.1
5. cuda 11.3
LLE-NET does not need special configurations. Just basic environment. 


### Folder structure
Download the LLE-NET code first.
The following shows the basic folder structure.
```
├────data/
│    └────result/
│    │    ├────con-4-40-15-5-8/
│    │    │    ├────DICM/
│    │    │    ├────Epoch19.pth
│    │    │    ├────LIME/
│    │    │    │    ├────1.bmp
│    │    │    │    ├────10.bmp
│    │    │    │    ├────2.bmp
│    │    │    │    ├────3.bmp
│    │    │    │    ├────4.bmp
│    │    │    │    ├────5.bmp
│    │    │    │    ├────6.bmp
│    │    │    │    ├────7.bmp
│    │    │    │    ├────8.bmp
│    │    │    │    └────9.bmp
│    │    │    ├────LOL_low/
│    │    │    │    └────LIME/
│    │    │    │    │    └────save_img/
│    │    │    ├────MEF/
│    │    │    ├────NPE/
│    │    │    └────VV/
│    │    ├────con_gamma_1/
│    │    │    ├────DICM/
│    │    │    ├────LIME/
│    │    │    │    ├────1.bmp
│    │    │    │    ├────10.bmp
│    │    │    │    ├────2.bmp
│    │    │    │    ├────3.bmp
│    │    │    │    ├────4.bmp
│    │    │    │    ├────5.bmp
│    │    │    │    ├────6.bmp
│    │    │    │    ├────7.bmp
│    │    │    │    ├────8.bmp
│    │    │    │    └────9.bmp
│    │    │    ├────LOL_low/
│    │    │    │    └────LIME/
│    │    │    │    │    └────save_img/
│    │    │    ├────MEF/
│    │    │    ├────NPE/
│    │    │    └────VV/
│    │    ├────DICM/
│    │    ├────LIME/
│    │    │    ├────1.bmp
│    │    │    ├────10.bmp
│    │    │    ├────2.bmp
│    │    │    ├────3.bmp
│    │    │    ├────4.bmp
│    │    │    ├────5.bmp
│    │    │    ├────6.bmp
│    │    │    ├────7.bmp
│    │    │    ├────8.bmp
│    │    │    └────9.bmp
│    │    ├────LOL_low/
│    │    │    └────LIME/
│    │    │    │    └────save_img/
│    │    ├────MEF/
│    │    ├────NPE/
│    │    └────VV/
├────dataloader.py
├────evaluate.py # 
├────lowlight_test.py
├────lowlight_test_gamma_itenum_1.py
├────lowlight_test_ite_num_1.py
├────lowlight_test_ite_num_2.py
├────lowlight_train.py
├────lowlight_train_gamma_itenum_1.py
├────lowlight_train_gamma_itenum_2.py
├────lowlight_train_ite_num_1.py
├────lowlight_train_ite_num_2.py
├────model.py
├────model_gamma_itenum_1.py
├────model_ite_num_1.py
├────model_ite_num_2.py
├────Myloss.py
├────Myloss_ite_num_1.py
├────Myloss_ite_num_2.py
├────README.md
└────snapshots/
│    └────Epoch19.pth
```
### Test: 

cd LLE-NET_code -upload
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
1) cd LLE-NET_code -upload

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```

## Contact
If you have any questions, please contact Xiujie Cao at caoxiujie@buaa.edu.cn.
