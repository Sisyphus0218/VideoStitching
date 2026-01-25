# README

## 环境配置

```bash
conda create --name video_stitching python=3.10
conda activate video_stitching
conda install numpy opencv tqdm 
```

## 文件夹结构

code 存放代码，data 存放数据，result 存放结果。

将待拼接的视频放在 `VideoStitching/data/videos` 中。

```
- VideoStitching
    - code
        - sift.py
        - stitch_images.py
        - stitch_videos.py
    - data
        - images
            - test.images1.png
            - test.images2.png
            ...
        - videos
            - 1.mp4
            - 2.mp4
            ...
    - result
```

## 运行方式

```bash
cd VideoStitching
python code/stitch_videos.py
```

## 代码说明

整体思路：

1. 利用视频的第一帧，提取 SIFT 特征并进行匹配，计算变换矩阵。
2. 后续帧直接复用第一帧的变换矩阵，直接进行变换和拼接。
3. 计算每两张图片之间匹配点的数目，选择匹配点数目最多的图片为起始图。起始图一般会在全景图片的中央，从图片中央向两边进行拼接，减轻累计误差。

### SIFT 特征检测器 `sift.py`

该模块实现了 `SiftDetector` 类，用于识别图片中的关键点，并匹配两张图片的关键点。

- `detect_and_compute`：对图像进行高斯模糊去噪，提取 SIFT 关键点及其描述符。
- `match_images`：使用 BFMatcher 寻找两幅图之间的对应点，并剔除距离过大的匹配对。

### 图像拼接器 `stitch_images.py`

该模块实现了 `ImageStitcher` 类，用于拼接多张图片。

- `read_images`：从图片数据文件夹读取图片。
- `build_match_graph`：记录每两张图片之间匹配点的数目。
- `find_stitch_order`：选择匹配点数目最多的图片为起始图，这张图片一般会在全景图片的中央，所以能减轻误差积累的问题。使用 Prim 算法计算拼接顺序。
- `compute_pairwise_homography`：计算两张图片之间的变换矩阵
- `compute_homographies`：通过矩阵乘法，计算各张图片相对于起始图的变换矩阵。
- `compute_canvas_params`：计算拼接后全景图的总尺寸，以及每张图片的变换矩阵，防止图像经变换后坐标出现负值。
- `create_weight_maps`：为每张图生成权重图，越靠近中心权重越高，边缘权重越低。
- `warp_and_blend`：将多张图片融合为一张全景图，使用加权平均法处理重叠区域，消除明显的拼接接缝。

### 视频拼接器 `stitch_videos.py`

该模块实现了 `VideoStitcher` 类，用于拼接视频。

- `read_videos`：从视频数据文件夹读取视频。
- `stitch_videos`：利用视频的第一帧计算单应性矩阵和权重图，然后读取所有视频的对应帧，复用第一帧计算出的单应性矩阵和权重图直接进行变换。

## 参考

[Overview | Image Stitching](https://www.youtube.com/watch?v=J1DwQzab6Jg&list=PL2zRqk16wsdp8KbDfHKvPYNGF2L-zQASc&index=1)

[Creating Panoramas with OpenCV Python Image Stitching](https://www.youtube.com/watch?v=Zs51cg4mb0k)

[AutoStitch拓展1——视频拼接问题 - 知乎](https://zhuanlan.zhihu.com/p/103276738)

[stitching_tutorial/docs/Stitching Tutorial.md at master · OpenStitching/stitching_tutorial](https://github.com/OpenStitching/stitching_tutorial/blob/master/docs/Stitching%20Tutorial.md)
