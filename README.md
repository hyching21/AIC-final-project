# AIC final project
## Run code
### require packages
```
pip install staintools
pip install spams-bin
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  --> 呃我不確定這個，可能就裝適合你的CUDA & pytorch
```
### data preparation
去kaggle 下載好資料後放到dataset 裡面，像這樣:
```
final_project
|__ dataset
|   |__ Grade.csv
|   |__ testA_1.bmp
|   |__ testA_1_anno.bmp
|   |__ testA_2.bmp
|   |__ testA_2_anno.bmp
|   |__ ...
|   |__ train_85.bmp
|   |__ train_85_anno.bmp
|__ prepare_dataset.py
|__ unet.py
```
然後跑`python prepare_dataset.py`，他會把資料整理成images、masks、test、train，變成這樣
```
final_project
|__ dataset
|   |__ Grade.csv
|   |__ images
|   |   |__ test
|   |   |   |__ testA_1.bmp
|   |   |   |__ testA_2.bmp
|   |   |   |__ ...
|   |   |   |__ testB_20.bmp
|   |   |
|   |   |__ train
|   |       |__ train_1.bmp
|   |       |__ train_2.bmp
|   |       |__ ...
|   |       |__ train85.bmp
|   |
|   |__ masks
|       |__ test
|       |   |__ testA_1_anno.bmp
|       |   |__ testA_2_anno.bmp
|       |   |__ ...
|       |   |__ testB_20_anno.bmp
|       |
|       |__ train
|           |__ train_1_anno.bmp
|           |__ train_2_anno.bmp
|           |__ ...
|           |__ train85_anno.bmp
|
|__ prepare_dataset.py
|__ unet.py
```
### train model
```
python unet.py
```
跑完會把模型存到model/unet.pth<br>
輸出圖片會存到results/{image name}_pred.bmp，image name 是圖片原本的名字(testA_1.bmp, testB_1.bmp ...)，也可以自己改line 215: filename.replace(".bmp", "**_pred**.bmp")) 改成自己想要的輸出名 (ex: filename.replace(".bmp", "**_myOutputName**.bmp")))<br><br>
[這裡](https://drive.google.com/drive/folders/1rVyESy9RzuWLcLJSpSTzRsND2tFs0J2E?usp=sharing)可以下載我之前train 好的模型(但我還沒辦法讓他可以直接load 存下來的model :(

## results
那個因為我code 還沒寫得很完整，所以要跑不同設定可能只能自己手動改code了，不能用args 直接指定 :( 搜哩
### 1. base (train 3 min, UNet_0511_base.pth)
預設是把圖片轉成灰階、resize to 256*256，跑30個epoch<br>
dice: 0.8491, IoU: 0.7492, last training loss: 1.7188
<p align="center">
<img src="example/testA_1.bmp" width="200"/>
<-- testA_1, testA_8-->
<img src="example/testA_8.bmp" width="200"/>
</p>

### 2. 增加epoch
- epoch = 300 (train 18 min, UNet_0511_ep300.pth)<br>
跟base 的設定一樣，只是epoch = 300<br>
line 17: epoch = 300<br>
dice: 0.8645, IoU: 0.7709, last training loss: 0.0675
<p align="center">
<img src="example/testA_1_pred.bmp" width="200"/>
<-- testA_1, testA_8-->
<img src="example/testA_8_pred.bmp" width="200"/>
</p>

### 3. 用彩色圖片去訓練 + color normalization
base + 用macenko 做color normalization，train_6.bmp 當 target img<br><br>
line 33: self.enc1 = CBR(**1**, 64) --> self.enc1 = CBR(**3**, 64)<br>
line 159: train_set = GlandDataset(train_imgs, train_masks) --> train_set = GlandDataset(train_imgs, train_masks, **target_img_path**)<br>
line 164: test_set = GlandDataset(test_imgs, test_masks) --> test_set = GlandDataset(test_imgs, test_masks, **target_img_path**)
- epoch = 30 (train 45 min, UNet_0512_macenko30.pth):<br>
dice: 0.7775, IoU: 0.6571, last training loss: 1.6895 (比gray scale差)
- epoch = 300 (train 7h 45min, UNet_0512_macenko300.pth):<br>
dice: 0.8791, IoU: 0.7927, last training loss: 0.0663
<p align="center">
<img src="example/testA_1_macenko.bmp" width="200"/>
<-- testA_1 (ep=30) testA_8 -->
<img src="example/testA_8_macenko.bmp" width="200"/>
<img src="example/testA_1_macenko300.bmp" width="200"/>
<-- testA_1 (ep=300) testA_8-->
<img src="example/testA_8_macenko300.bmp" width="200"/>
</p>

### 4. resize + padding
灰階，把圖片較短的邊resize 到256，另一邊等比例縮小(base 是直接把圖片變成正方形去訓練，這裡保留原本的長寬比)，不夠的地方zero padding，最後整張圖片還是256*256(為了配合UNet 的架構，他好像只能收$2^n$ * $2^n$的圖片?)<br><br>
註解掉: line 117, 122, 130, 135<br>
取消註解: line 116, 121, 129, 134
- epoch = 30 (train 2 min, UNet_0514_resize30.pth)<br>
dice: 0.8367, IoU: 0.7343 (比gray scale差)
- epoch = 300 (train 18 min, UNet_0514_resize300.pth)<br>
dice: 0.8531, IoU: 0.7571, last training loss: 0.0600 (好像沒有很好?)
<p align="center">
<img src="example/testA_1_resize.bmp" width="200"/>
<-- testA_1 (ep=30) testA_8 -->
<img src="example/testA_8_resize.bmp" width="200"/>
<img src="example/testA_1_resize300.bmp" width="200"/>
<-- testA_1 (ep=300) testA_8-->
<img src="example/testA_8_resize300.bmp" width="200"/>
</p>

## TODO
- 修正圖片輸出
    > 現在的輸出圖片是類似灰階mask，只有用黑白表示，希望可以是輸出原本的彩色圖片，用線框出有問題的區域
- 用其他模型做預測
    > DCAN?
- 更多實驗
    > try different ep, lr.<br>
    > different color normalization method: vahadane
    > different color normalization target image (current train_6)
    > how to improve rgb image's prediction result? 為什麼用彩色圖片訓練反而比灰階的訓練結果還要差!?
    > training set & validation set
    > try different UNet architecture?
