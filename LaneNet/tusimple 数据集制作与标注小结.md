# tusimple 格式数据集制作与标注小结

最近在看 LaneNet 然后根据网上的博客写下一些自己在制作数据集中的过程。

## 1.前期准备

工具： labelme 3.6.12  + windows10

(1) 先创建一个虚拟环境，控制台指令：

```python
conda  create -n yourenvname python==3.6.0
```

(2) 激活环境并安装，控制台指令：(推荐用国内源)

```
conda activate yourenvname
pip install labelme==3.6.12
```

## 2.进行标注

(1) 控制台进入软件：

直接控制台环境下输入 labelme ；

(2) 利用点(Point)进行标注(亲测可用)：

控制台进入labelme, Edit $\rightarrow$Create Point , 即可在图像上进行标注，双击生成label并输入；

(3) 利用线(LineStrip)进行标注(亲测可用)：

控制台进入labelme, Edit $\rightarrow$Create LineStrip , 即可在图像上进行标注，双击生成label并输入；

最后得到一系列 .json 文件

![](D:\Study\markdown_pic\tusimple\json文件截图.jpg)

**PS**：笔者暂时只做到利用 LaneNet 可以进行正常的训练，不知道是否有好的效果。

## 3.数据格式转化

(1) 将 json 文件转化成 dataset

```python
import argparse
import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml
from labelme import utils
import base64

# 存在一个需要优化的 BUG ，就是只能将文件写到默认的文件所在
# 的目录下，这点需要优化
# 批量转换代码

def main():

    warnings.warn("This script is aimed to demonstrate how to convert the\n"

                  "JSON file to a single image dataset, and not to handle\n"

                  "multiple JSON files to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
	# 只要把这个路径改成你的 json 文件所在的路径就可以
    parser.add_argument('--json_file', default='D:\\Study\\Video_Frame\\Line\\')
    # parser.add_argument('json_file')
	
    parser.add_argument('--out', default=None)
    # parser.add_argument('--out', default="D:/Study/Video_Frame/Points_out/")

    args = parser.parse_args()

    json_file = args.json_file

    if args.out is None:

        out_dir = osp.basename(json_file).replace('.', '_')

        out_dir = osp.join(osp.dirname(json_file), out_dir)

    else:

        out_dir = args.out

    if not osp.exists(out_dir):

        os.mkdir(out_dir)

    count = os.listdir(json_file)

    for i in range(0, len(count)):

        path = os.path.join(json_file, count[i])

        if os.path.isfile(path):

            data = json.load(open(path))


            if data['imageData']:

                imageData = data['imageData']

            else:

                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])

                with open(imagePath, 'rb') as f:

                    imageData = f.read()

                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {'_background_': 0}

            for shape in data['shapes']:

                label_name = shape['label']

                if label_name in label_name_to_value:

                    label_value = label_name_to_value[label_name]

                else:

                    label_value = len(label_name_to_value)

                    label_name_to_value[label_name] = label_value


            # label_values must be dense

            label_values, label_names = [], []

            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):

                label_values.append(lv)

                label_names.append(ln)

            assert label_values == list(range(len(label_values)))


            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            captions = ['{}: {}'.format(lv, ln)

                for ln, lv in label_name_to_value.items()]

            lbl_viz = utils.draw_label(lbl, img, captions)


            out_dir = osp.basename(count[i]).replace('.', '_')

            out_dir = osp.join(osp.dirname(count[i]), out_dir)

            if not osp.exists(out_dir):

                os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))

            #PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))

            utils.lblsave(osp.join(out_dir, 'label.png'), lbl)

            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:

                for lbl_name in label_names:

                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')

            info = dict(label_names=label_names)

            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:

                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)

if __name__ == '__main__':

    main()

```

以Create Strip 标注的数据集生成结果如下，用Create Point 步骤完全一样。

![](D:\Study\markdown_pic\tusimple\数据集制作.jpg)

(2) dataset 转 tusimple 数据集格式

```python
# data.py
import cv2
from skimage import measure, color
from skimage.measure import regionprops
import numpy as np
import os
import copy

# 优化的地方：应该用文件的名字来命名新的文件，而不是简单的进行 +1 操作就进行对图片文件的命名
# BUG 已解决
def skimageFilter(gray):
    binary_warped = copy.copy(gray)
    binary_warped[binary_warped > 0.1] = 255
    gray = (np.dstack((gray, gray, gray)) * 255).astype('uint8')
    labels = measure.label(gray[:, :, 0], connectivity=1)
    dst = color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    gray = cv2.cvtColor(np.uint8(dst * 255), cv2.COLOR_RGB2GRAY)
    return binary_warped, gray

def moveImageTodir(path, targetPath, name):
    if os.path.isdir(path):
        image_name = "gt_image/" + str(name) + ".png"
        binary_name = "gt_binary_image/" + str(name) + ".png"
        instance_name = "gt_instance_image/" + str(name) + ".png"
        train_rows = image_name + " " + binary_name + " " + instance_name + "\n"
        origin_img = cv2.imread(path + "/img.png")
        origin_img = cv2.resize(origin_img, (1280, 720))
        cv2.imwrite(targetPath + "/" + image_name, origin_img)
        img = cv2.imread(path + '/label.png')
        img = cv2.resize(img, (1280, 720))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_warped, instance = skimageFilter(gray)

        cv2.imwrite(targetPath + "/" + binary_name, binary_warped)
        cv2.imwrite(targetPath + "/" + instance_name, instance)

        print("success create data name is : ", train_rows)
        return train_rows
    return None

if __name__ == "__main__":
    print('--------------开始执行----------------')
    #count = 1
    with open("./train.txt", 'w+') as file:
        for images_dir in os.listdir("./images_line"):
            dir_name = os.path.join("./images_line", images_dir + "/annotations")
            for annotations_dir in os.listdir(dir_name):
                json_dir = os.path.join(dir_name, annotations_dir)
                if os.path.isdir(json_dir):
                    # train_rows = moveImageTodir(json_dir, "./", str(count).zfill(4))
                    # 利用 json 文件夹的名字来对生成的图片进行命名，比较好对应
                    train_rows = moveImageTodir(json_dir, "./", json_dir.split('\\')[-1])
                    file.write(train_rows)
                    #count += 1
```

生成结果如下：

以 gt_binary_image 为例：

<img src="D:\Study\markdown_pic\tusimple\199_json.png" style="zoom:50%">

通过上面的操作就可以得到 tusimple 数据集格式的个人数据集，就可以放到 LaneNet 中去训练啦，可以参考我的另一篇博客：[LaneNet调试记录](https://blog.csdn.net/yyq_163/article/details/119829158)

## 参考博客：

[创建自己的 tusimple 数据集格式](https://blog.csdn.net/qq_26894673/article/details/100052191)

[制作 tusimple 数据集格式的数据](https://blog.csdn.net/weixin_42398658/article/details/113941958)

[tusimple车道线检测 处理自己的数据集 用自己的数据集训练模型](https://blog.csdn.net/qq_40900196/article/details/103600292)

[车道线识别  tusimple 数据集介绍](https://blog.csdn.net/qingfengxd1/article/details/108738651#comments_18028312)

[Labelme标注的数据转换为tusimple数据集格式](https://www.jianshu.com/p/6cd7c42da802)



此博客仅供学习交流，如有侵权请联系我删除，谢谢。