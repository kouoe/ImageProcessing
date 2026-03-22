import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.fft import fft2, fftshift, ifft2, ifftshift
#设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def create_f1():
"""创建f1图像"""
f1 = np.zeros((256, 256))
#中间亮条：128×32，位置在中心
start_row = (256-128) // 2
end_row = start_row + 128
start_col = (256-32) // 2
end_col = start_col + 32
f1[start_row:end_row, start_col:end_col] = 100
return f1
def plot_spectrum(fft_result, title):
"""绘制频谱图"""
magnitude_spectrum = np.log(1 + np.abs(fft_result))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title(title)
plt.colorbar()
def gaussian_lp_filter(shape, d0):
"""高斯低通滤波器"""
m, n = shape
u = np.arange(m)
v = np.arange(n)
u, v = np.meshgrid(u, v)
d = np.sqrt((u-m/2)**2 + (v-n/2)**2)
h = np.exp(-(d**2) / (2 * d0**2))
return h
def gaussian_hp_filter(shape, d0):
"""高斯高通滤波器"""
return 1-gaussian_lp_filter(shape, d0)
def butterworth_lp_filter(shape, d0, n=2):
"""巴特沃斯低通滤波器"""
m, n = shape
u = np.arange(m)
v = np.arange(n)
u, v = np.meshgrid(u, v)
d = np.sqrt((u-m/2)**2 + (v-n/2)**2)
h = 1 / (1+ (d / d0)**(2*n))
return h
def butterworth_hp_filter(shape, d0, n=2):
"""巴特沃斯高通滤波器"""
return 1-butterworth_lp_filter(shape, d0, n)
def load_and_preprocess_image(image_path):
"""加载并预处理图像"""
#读取图像
img = cv2.imread(image_path)
#转换为灰度图
if len(img.shape) == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#调整大小为256x256（可选，根据你的需求）
img = cv2.resize(img, (256, 256))
#归一化到0-255
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
return img.astype(np.float64)
#
=======================================================================
=====
#问题1：图像变换和频域处理实验
#
=======================================================================
=====
print("开始问题1：图像变换和频域处理实验")
# 1.1创建f1图像并显示
f1 = create_f1()
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(f1, cmap='gray')
plt.title('1.1 f1原图')
# 1.2对f1进行FFT
F1 = fft2(f1)
F1_shift = fftshift(F1)
plt.subplot(2, 3, 2)
plot_spectrum(F1_shift, '1.2 FFT(f1)幅度谱')
# 1.3创建f2并FFT
m, n = f1.shape
x, y = np.meshgrid(np.arange(m), np.arange(n))
f2 = ((-1) ** (x + y)) * f1
plt.subplot(2, 3, 3)
plt.imshow(f2, cmap='gray')
plt.title('1.3 f2 = (-1)^(m+n) * f1')
F2 = fft2(f2)
F2_shift = fftshift(F2)
plt.subplot(2, 3, 4)
plot_spectrum(F2_shift, '1.3 FFT(f2)幅度谱')
# 1.4旋转f2得到f3并FFT
f3 = ndimage.rotate(f2,-45, reshape=False, order=1) #顺时针旋转45度
plt.subplot(2, 3, 5)
plt.imshow(f3, cmap='gray')
plt.title('1.4 f3 =旋转45°的f2')
F3 = fft2(f3)
F3_shift = fftshift(F3)
plt.subplot(2, 3, 6)
plot_spectrum(F3_shift, '1.4 FFT(f3)幅度谱')
plt.tight_layout()
plt.show()
# 1.5旋转f1得到f4，创建f5并FFT
plt.figure(figsize=(15, 8))
f4 = ndimage.rotate(f1,-90, reshape=False, order=1) #顺时针旋转90度
f5 = f1 + f4
plt.subplot(2, 3, 1)
plt.imshow(f4, cmap='gray')
plt.title('1.5 f4 =旋转90°的f1')
plt.subplot(2, 3, 2)
plt.imshow(f5, cmap='gray')
plt.title('1.5 f5 = f1 + f4')
F4 = fft2(f4)
F4_shift = fftshift(F4)
F5 = fft2(f5)
F5_shift = fftshift(F5)
plt.subplot(2, 3, 3)
plot_spectrum(F4_shift, '1.5 FFT(f4)幅度谱')
plt.subplot(2, 3, 4)
plot_spectrum(F5_shift, '1.5 FFT(f5)幅度谱')
# 1.6创建f6并FFT
f6 = f2 + f3
F6 = fft2(f6)
F6_shift = fftshift(F6)
plt.subplot(2, 3, 5)
plt.imshow(f6, cmap='gray')
plt.title('1.6 f6 = f2 + f3')
plt.subplot(2, 3, 6)
plot_spectrum(F6_shift, '1.6 FFT(f6)幅度谱')
plt.tight_layout()
plt.show()
#
=======================================================================
=====
#问题2：频域滤波实验-对同一张加噪声的图像进行五种滤波处理
#
=======================================================================
=====
print("开始问题2：频域滤波实验")
#请将下面的路径替换为你自己的图像路径
image_path = "D:/cv/adc.png" #请替换为实际路径
#加载图像
try:
original_image = load_and_preprocess_image(image_path)
#添加噪声
noise = np.random.normal(0, 20, original_image.shape)
noisy_image = original_image + noise
noisy_image = np.clip(noisy_image, 0, 255)
#对噪声图像进行FFT（对应第二题第1小问）
F_noisy = fft2(noisy_image)
F_noisy_shift = fftshift(F_noisy)
#设置不同的D0值
D0_values = [10, 20, 40, 80]
#显示原图和噪声图像
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原图')
plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声图像')
plt.subplot(1, 3, 3)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
# 2.2高斯低通滤波
plt.figure(figsize=(16, 12))
plt.suptitle('2.2高斯低通滤波', fontsize=16)
for i, d0 in enumerate(D0_values):
#高斯低通滤波
gaussian_lp = gaussian_lp_filter(noisy_image.shape, d0)
F_filtered = F_noisy_shift * gaussian_lp
filtered_image = np.real(ifft2(ifftshift(F_filtered)))
plt.subplot(3, 4, i+1)
plt.imshow(gaussian_lp, cmap='gray')
plt.title(f'高斯低通滤波器D0={d0}')
plt.subplot(3, 4, i+5)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'滤波后图像D0={d0}')
plt.subplot(3, 4, 9)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声原图')
plt.subplot(3, 4, 10)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
# 2.3高斯高通滤波
plt.figure(figsize=(16, 12))
plt.suptitle('2.3高斯高通滤波', fontsize=16)
for i, d0 in enumerate(D0_values):
#高斯高通滤波
gaussian_hp = gaussian_hp_filter(noisy_image.shape, d0)
F_filtered = F_noisy_shift * gaussian_hp
filtered_image = np.real(ifft2(ifftshift(F_filtered)))
plt.subplot(3, 4, i+1)
plt.imshow(gaussian_hp, cmap='gray')
plt.title(f'高斯高通滤波器D0={d0}')
plt.subplot(3, 4, i+5)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'滤波后图像D0={d0}')
plt.subplot(3, 4, 9)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声原图')
plt.subplot(3, 4, 10)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
# 2.4巴特沃斯低通滤波
plt.figure(figsize=(16, 12))
plt.suptitle('2.4巴特沃斯低通滤波(n=2)', fontsize=16)
for i, d0 in enumerate(D0_values):
#巴特沃斯低通滤波
butterworth_lp=butterworth_lp_filter(noisy_image.shape,d0,n=2)
F_filtered = F_noisy_shift * butterworth_lp
filtered_image = np.real(ifft2(ifftshift(F_filtered)))
plt.subplot(3, 4, i+1)
plt.imshow(butterworth_lp, cmap='gray')
plt.title(f'巴特沃斯低通D0={d0}')
plt.subplot(3, 4, i+5)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'滤波后图像D0={d0}')
plt.subplot(3, 4, 9)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声原图')
plt.subplot(3, 4, 10)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
# 2.5巴特沃斯高通滤波
plt.figure(figsize=(16, 12))
plt.suptitle('2.5巴特沃斯高通滤波(n=2)', fontsize=16)
for i, d0 in enumerate(D0_values):
#巴特沃斯高通滤波
butterworth_hp=butterworth_hp_filter(noisy_image.shape,d0,n=2)
F_filtered = F_noisy_shift * butterworth_hp
filtered_image = np.real(ifft2(ifftshift(F_filtered)))
plt.subplot(3, 4, i+1)
plt.imshow(butterworth_hp, cmap='gray')
plt.title(f'巴特沃斯高通D0={d0}')
plt.subplot(3, 4, i+5)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'滤波后图像D0={d0}')
plt.subplot(3, 4, 9)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声原图')
plt.subplot(3, 4, 10)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
except Exception as e:
print(f"处理图像{image_path}时出错: {e}")
#

#结果分析和比较
#


print("生成结果分析和比较图...")
#比较不同滤波器在相同D0下的效果
d0_comparison = 20
try:
plt.figure(figsize=(16, 10))
plt.suptitle(f'不同滤波器比较(D0={d0_comparison})', fontsize=16)
#原图
plt.subplot(2, 4, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('加噪声原图')
#高斯低通
gaussian_lp = gaussian_lp_filter(noisy_image.shape, d0_comparison)
F_gaussian_lp = F_noisy_shift * gaussian_lp
img_gaussian_lp = np.real(ifft2(ifftshift(F_gaussian_lp)))
plt.subplot(2, 4, 2)
plt.imshow(img_gaussian_lp, cmap='gray')
plt.title('高斯低通')
#高斯高通
gaussian_hp = gaussian_hp_filter(noisy_image.shape, d0_comparison)
F_gaussian_hp = F_noisy_shift * gaussian_hp
img_gaussian_hp = np.real(ifft2(ifftshift(F_gaussian_hp)))
plt.subplot(2, 4, 3)
plt.imshow(img_gaussian_hp, cmap='gray')
plt.title('高斯高通')
#巴特沃斯低通
butterworth_lp = butterworth_lp_filter(noisy_image.shape,
d0_comparison, n=2)
F_butterworth_lp = F_noisy_shift * butterworth_lp
img_butterworth_lp = np.real(ifft2(ifftshift(F_butterworth_lp)))
plt.subplot(2, 4, 4)
plt.imshow(img_butterworth_lp, cmap='gray')
plt.title('巴特沃斯低通')
#巴特沃斯高通
butterworth_hp = butterworth_hp_filter(noisy_image.shape,
d0_comparison, n=2)
F_butterworth_hp = F_noisy_shift * butterworth_hp
img_butterworth_hp = np.real(ifft2(ifftshift(F_butterworth_hp)))
plt.subplot(2, 4, 5)
plt.imshow(img_butterworth_hp, cmap='gray')
plt.title('巴特沃斯高通')
#显示滤波器
plt.subplot(2, 4, 6)
plt.imshow(gaussian_lp, cmap='gray')
plt.title('高斯低通滤波器')
plt.subplot(2, 4, 7)
plt.imshow(butterworth_lp, cmap='gray')
plt.title('巴特沃斯低通滤波器')
plt.subplot(2, 4, 8)
plot_spectrum(F_noisy_shift, '噪声图像频谱')
plt.tight_layout()
plt.show()
except Exception as e:
print(f"比较图像时出错: {e}")
print("所有实验完成！")
#

#结果分析和总结
#

print("\n===实验结果分析===")
print("\n问题1总结：")
print("1. FFT(f1)和FFT(f2)：f2通过(-1)^(m+n)预处理，使频谱中心化")
print("2. FFT(f2)和FFT(f3)：空域旋转45°对应频域也旋转45°")
print("3. FFT(f5) = FFT(f1) + FFT(f4)，体现傅里叶变换的线性性质")
print("4. FFT(f6) = FFT(f2) + FFT(f3)，同样体现线性性质")
print("\n问题2总结：")
print("1.低通滤波：D0越小，图像越模糊，噪声抑制越强")
print("2.高通滤波：D0越小，边缘越突出，细节增强")
print("3.高斯滤波器过渡平滑，巴特沃斯滤波器有更明确的截止特性")
print("4.相同D0下，高斯滤波比巴特沃斯滤波效果更柔和")
print("5.低通滤波能有效去除噪声但会模糊图像，高通滤波能增强边缘但会保留噪声"）