import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def check_image_path(img_path):
    """检查图像路径是否存在"""
    if not os.path.exists(img_path):
        print(f"错误：图像文件 '{img_path}' 不存在")
        return False
    return True

def task1_image_blending(img_path1, img_path2):
    """任务1：图像叠加融合"""
    # 检查图像文件是否存在
    if not check_image_path(img_path1) or not check_image_path(img_path2):
        return None
    
    try:
        # 读取图像
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            print("无法读取图像文件")
            return None
        
        # 调整图像尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转换为浮点数便于计算
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # 测试不同权重组合
        weight_combinations = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]
        
        results = []
        for c1, c2 in weight_combinations:
            blended = c1 * img1 + c2 * img2
            results.append((c1, c2, blended))
        
        return results
    except Exception as e:
        print(f"任务1执行错误: {e}")
        return None

def task2_image_enhancement(img_path):
    """任务2：骨骼图像增强流程"""
    # 检查图像文件是否存在
    if not check_image_path(img_path):
        return None
    
    try:
        # 读取原图并转换为灰度图
        img_a = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_a is None:
            print(f"无法读取图像文件: {img_path}")
            return None
            
        img_a = img_a.astype(np.float32) / 255.0
        
        # (b) 拉普拉斯变换
        kernel_laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        img_b = cv2.filter2D(img_a, -1, kernel_laplace)
        
        # (c) 原图与拉普拉斯图相加（锐化）
        img_c = img_a + img_b
        img_c = np.clip(img_c, 0, 1)
        
        # (d) Sobel边缘检测
        sobel_x = cv2.Sobel(img_a, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_a, cv2.CV_32F, 0, 1, ksize=3)
        img_d = np.sqrt(sobel_x**2 + sobel_y**2)
        img_d = img_d / np.max(img_d) if np.max(img_d) > 0 else img_d
        
        # (e) 5×5均值滤波平滑Sobel图像
        kernel_mean = np.ones((5,5), np.float32) / 25
        img_e = cv2.filter2D(img_d, -1, kernel_mean)
        
        # (f) 锐化图像与平滑Sobel图像相乘（掩蔽）
        img_f = img_c * img_e
        img_f = img_f / np.max(img_f) if np.max(img_f) > 0 else img_f
        
        # (g) 原图与掩蔽图像相加
        img_g = img_a + img_f
        img_g = np.clip(img_g, 0, 1)
        
        # (h) 幂律变换（伽马校正）
        gamma = 0.5
        img_h = np.power(img_g, gamma)
        
        # 返回所有处理结果
        results = {
            'a': img_a, 'b': img_b, 'c': img_c, 'd': img_d,
            'e': img_e, 'f': img_f, 'g': img_g, 'h': img_h
        }
        
        return results
    except Exception as e:
        print(f"任务2执行错误: {e}")
        return None

def display_results(results, title):
    """显示处理结果"""
    if results is None:
        print("无法显示结果：数据为空")
        return
        
    plt.figure(figsize=(15, 10))
    
    if isinstance(results, list):  # 任务1结果
        for i, (c1, c2, img) in enumerate(results):
            plt.subplot(2, 2, i+1)
            if len(img.shape) == 3 and img.shape[2] == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(f'c1={c1}, c2={c2}')
            plt.axis('off')
    else:  # 任务2结果
        titles = ['(a)原图', '(b)拉普拉斯', '(c)锐化图像', '(d)Sobel',
                 '(e)平滑Sobel', '(f)掩蔽图像', '(g)锐化结果', '(h)最终结果']
        
        for i, (key, title_text) in enumerate(zip('abcdefgh', titles)):
            plt.subplot(2, 4, i+1)
            plt.imshow(results[key], cmap='gray')
            plt.title(title_text)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 任务1：图像叠加
    print("执行任务1：图像叠加融合")
    results1 = task1_image_blending('bottle1.jpg', 'bottle2.jpg')
    if results1:
        display_results(results1, "图像叠加结果")
    
    # 任务2：图像增强
    print("执行任务2：骨骼图像增强")
    results2 = task2_image_enhancement('bone.png')
    if results2:
        display_results(results2, "骨骼图像增强流程")