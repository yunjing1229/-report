# 实验三：图像特征匹配与拼接参考文档

## 一、实验目的
学会使用 OpenCV 库中的 SIFT 特征提取与 FLANN 匹配器进行图像特征点匹配，利用 RANSAC 算法计算单应矩阵（Homography），最终实现两张具有重叠区域图像的自动拼接。

## 二、实验内容
上传原始图片

<img width="1280" height="720" alt="231624727c1df725f52d88f7cf0bdc5" src="https://github.com/user-attachments/assets/6b186cdf-c86f-485b-a0b6-e2c3a33416f5" />

<img width="1280" height="720" alt="5a3c0f63822329762d0750ae80eb3ea" src="https://github.com/user-attachments/assets/225b5401-6140-45f8-8dc2-f46ac6cdb792" />



### 2.1 导入图像拼接相关的依赖包

OpenCV库：核心计算机视觉库，提供 SIFT 算法、匹配器、图像变换等 API。
NumPy库：科学计算基础库，用于矩阵运算和数组重构。

```
import cv2
import numpy as np
```

### 2.2 读取原始图像并提取特征点
读取两张用于拼接的图像。为了减少计算量并提高特征提取的稳定性，我们将原图转换为灰度图。接着，初始化 SIFT 检测器，分别计算两张图的特征点和特征描述符。

```
# 读取图像
img_a = cv2.imread("a1.jpg")
img_b = cv2.imread("p1.jpg")

# 将图像转换为灰度图 (SIFT 算法基于灰度图工作)
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 初始化 SIFT 检测器并提取特征
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None) # 图像a的关键点和描述符
kp_b, des_b = sift.detectAndCompute(gray_b, None) # 图像b的关键点和描述符
```

### 2.3 特征匹配与筛选优质点
使用 FLANN（Fast Library for Approximate Nearest Neighbors）匹配器对描述符进行匹配。为了排除错误的匹配，使用 Lowe's Ratio Test（阈值取 0.7）。只有当第一匹配的距离显著小于第二匹配距离时，才认为是一个可靠的匹配点。

```
# 使用 FLANN 匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2) # k=2表示每个特征点返回2个最佳匹配

# 应用 Lowe's 比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 绘制匹配的 SIFT 关键点
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

### 2.4 估计单应矩阵与计算拼接画布大小

提取筛选后的匹配点坐标，使用 RANSAC 算法 估计透视变换矩阵（单应矩阵 H）。由于拼接后的图像尺寸会变大，且可能有部分像素落在负坐标区域，因此需要计算所有角点变换后的边界，以确定最终的拼接图像大小和所需的平移量。

```
# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用 RANSAC 算法估计单应矩阵（透视变换矩阵）
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像b变换后的四个角点坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸（包含所有像素）
all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

```


### 2.5 透视变换与图像融合
创建一个平移矩阵，将变换后可能为负的坐标平移到正数区域。利用组合矩阵对图像 b 进行透视变换，然后将图像 a 覆盖到对应的位置，完成拼接。

```
# 创建平移矩阵，确保所有像素都在可见区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 对图像b进行透视变换和平移
stitched_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H, # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min) # 输出图像尺寸
)

# 将图像a复制到拼接结果的对应位置
stitched_img[-y_min : h_a - y_min, -x_min : w_a - x_min] = img_a
```

## 三、实验结果与分析


<img width="1849" height="1174" alt="a25a7629506ba3325cc835576cf5a0a" src="https://github.com/user-attachments/assets/417dac8f-7a7a-4f93-8b62-f320a6022210" />
<img width="1881" height="1179" alt="6283b216fd847c508cf99fb53a154bb" src="https://github.com/user-attachments/assets/eb61486e-ed37-4ece-bb64-37089ff1de2c" />

### 3.1 特征点匹配结果（Matched SIFT Keypoints）
由于本次测试使用的是二次元插画图像，图像中存在大量平滑色块和相似线条。从匹配图可以看出，虽然部分特征点找到了准确的对应关系，但也由于特征高度重复，导致出现了较多跨越屏幕的误匹配连线。

### 3.2 图像拼接结果（Stitched Result）
尽管特征匹配环节较为杂乱，但 RANSAC 算法发挥了关键作用。它通过随机抽样和内点统计，成功剔除了所有的误匹配点，计算出高精度的透视变换矩阵。最终拼接出的图像中，人物脸部、发丝边缘、衣服轮廓以及高脚杯的线条完全对齐，没有任何重影和错位，且通过坐标平移成功消除了变换带来的黑边。

### 3.3 技术原理解析
SIFT 特征：对尺度缩放和旋转具有不变性，适合提取图像中的稳定特征点。

RANSAC 算法：在包含大量错误数据（Outliers）的情况下，依然能准确估算出数学模型（单应矩阵）。

画布平移：透视变换后的图像坐标会出现负数。直接拼接会导致部分图像被裁剪，通过 translation_matrix 将图像平移到正数坐标系，保证了画面完整。

## 四、实验小结
本实验完成了基于 SIFT 算法的图像特征提取、FLANN 特征匹配、RANSAC 单应矩阵估计以及最终的透视变换拼接。

实验结果表明：

SIFT + FLANN + Lowe's Ratio Test 能够提供较稳定的匹配点对，但在平滑色块较多的图像上仍会产生误匹配。

RANSAC 是实现图像对齐的核心，它能从杂乱的匹配点中找出真正的对应关系。

在图像拼接等计算机视觉任务中，处理图像的坐标平移和边界计算至关重要，否则容易导致拼接结果丢失。

通过本次实验，进一步加深了对特征匹配、透视变换、图像融合原理的理解，掌握了将两幅不同视角图像拼接成全景图的基本方法，为后续的计算机视觉高级应用奠定了基础。
