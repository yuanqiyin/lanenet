# windows10 下配置 LaneNet + 调试 Bug 记录

## 1. 环境准备

windows10 + Anaconda + cuda9.0 + tensorflow_gpu1.12.0 + cudnn7.4.1.5  + python3.5

显卡：1660ti (6G)

CUDA 下载官方网址：

首先要选择好对应版本，CUDA + cudnn，如果要使用 pytorch 或者 tensorflow 的话还要将三者的版本信息都要对应好。

CUDA 官方下载地址 : https://developer.nvidia.com/cuda-toolkit-archive

cudnn 官方下载地址 : https://developer.nvidia.com/rdp/cudnn-archive

(cudnn下载时需注册账号，QQ邮箱即可)

## 2. 数据准备

tusimple 数据集，GitHub 地址：https://github.com/TuSimple/tusimple-benchmark/issues/3

LaneNet 项目地址：https://github.com/MaybeShewill-CV/lanenet-lane-detection/

(下载速度慢的话需科学上网)

## 3. 安装环境

下面的安装推荐先换源，我是用的清华源，换源的博客有很多，百度即可，不再赘述。

(1) 首先创建虚拟环境:

```python
conda create -n yourenvironmentname python==3.5
```

(2) 安装项目所需的文件：

```python
pip install -r requirements.txt
```

## 4. 处理数据

(1) 生成train.txt：

```
python tools/generate_tusimple_dataset.py --src_dir 数据集解压的路径，例(D:\DataSet\yinyuanqi\tusimple)
```

上述操作过后会生成一个 train.txt 文件，文件内格式为上述路径 + 图片名称.png

(2)  生成 val.txt ,test.txt 以及 将标注格式转化成 tfrecords:

```
python data_provider/lanenet_data_feed_pipline.py 
--dataset_dir D:\\yinyuanqi\\DataSet\\tusimple\\training\\
--tfrecords_dir ./data/training_data_example/tfrecords
```

--dataset_sir 是你数据集存放的位置下面的 training 文件夹，既可以像上面一样在控制台直接给出，也可以在 lanenet_data_feed_pipline.py 文件中做调整；

--tfrecords_dir  是生成 tfrecords 的位置

下图是我做完上述两个步骤的文件夹格式：

## ![1629452084721](C:\Users\远齐\AppData\Roaming\Typora\typora-user-images\1629452084721.png)5.测试代码

直接运行测试文件 test_lanenet.py, 下面给出控制台指令示例：

```python
python tools/test_lanenet.py
 --weights_path model/New_Tusimple_Lanenet_Model_Weights/tusimple_lanenet_vgg.ckpt
 --image_path data/tusimple_test_image/0.jpg
```

若遇到找不到模块的错误，我是将测试文件直接移动到根目录下进行执行，相应改下上面的指令即可。

## 6.训练代码

训练控制台指令：

```
python train_lanenet.py
 --net vgg
 --dataset_dir D:\\Project\\lanenet-lane-detection-master\\data\\training_example\\
 -m 0
```

这里需要下载一个文件： vgg.npy,在下面的参考博客中，好心博主已给出百度云链接，自取即可。

## Bug 记录

(1) 版本问题

```
Error : Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
```

我是将原本作者用的cudnn7.0 换成了cudnn7.4.1.5

参考： https://github.com/tensorflow/tensorflow/issues/24828

有博主说这样的错误是因为GPU显存问题，我也同时参考：

https://www.jianshu.com/p/31e377585318

https://blog.csdn.net/tsyccnh/article/details/102938368

(2)  路径问题

```
cv2.error: OpenCV(4.0.0) C:\projects\opencv-python\opencv\modules\imgproc\src\resize.cpp:3784: error: (-215:Assertion failed) !ssize.empty() in functi on 'cv::resize'
```

(3) 在重新生成train.txt,val.txt,test.txt 之前要将原来的旧文件删除，否则会报错，报错信息为：

```
AttributeError: 'NoneType' object has no attribute 'shape'
```

还要删除相应的 gt_image, gt_image_binary, gt_image_instance 文件夹里的不相符的图片，然后再生成tfrecords，tfrecords是根据上面三个文件夹中的图片生成的，也相当于是 tensorflow 里面的驱动

要删除相应的文件并重新生成，然后进行训练，训练后会得到 .ckpt 文件，不需要的也要删除

(4) 遇到 BUG 先检查自己的路径，如果是相对路径就先改成绝对路径试试，有时会有惊喜

BUG记录不定期更新

## 参考博客：

我写的只针对于我的环境，下面给出的参考博客写的非常好，可以进行参考：

参考博客：https://blog.csdn.net/qq_40900196/article/details/102840850

​					https://blog.csdn.net/qq_41821608/article/details/113582019

## 小 Tips:

Win10系统下不同版本cuda切换(利用系统调用环境变量的顺序来做的)

https://blog.csdn.net/sinat_38132146/article/details/106252877

**最后欢迎大家一起交流学习！**