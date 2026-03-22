import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ==================== 图像文件检查和处理函数 ====================

def check_image_exists(base_path, image_name):
    """检查单个图像文件是否存在"""
    img_path = os.path.join(base_path, image_name)
    if os.path.exists(img_path):
        return True, img_path
    else:
        print(f"警告: 图像文件 {img_path} 不存在")
        return False, img_path

def preprocess_image(image_path):
    """图像预处理：转为灰度图并降低灰度级到16级"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像（可能格式不支持）: {image_path}")
            return None
        
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 将256级灰度转换为16级（0-255映射到0-15）
        gray_16 = (gray // 16).astype(np.uint8)
        
        return gray_16
    except Exception as e:
        print(f"预处理图像时出错 {image_path}: {e}")
        return None

# ==================== 手动实现GLCM和纹理特征 ====================

def calculate_glcm_manual(image, distance=1, angle=0, levels=16):
    """手动计算灰度共生矩阵"""
    try:
        height, width = image.shape
        glcm = np.zeros((levels, levels), dtype=np.float32)
        
        # 根据角度计算偏移量
        if angle == 0:  # 0度
            dx, dy = distance, 0
        elif angle == 45:  # 45度
            dx, dy = distance, -distance
        elif angle == 90:  # 90度
            dx, dy = 0, -distance
        elif angle == 135:  # 135度
            dx, dy = -distance, -distance
        else:
            raise ValueError("角度必须是0, 45, 90或135度")
        
        # 计算GLCM
        y_start = max(0, -dy)
        y_end = min(height, height - dy)
        x_start = max(0, -dx)
        x_end = min(width, width - dx)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                i = image[y, x]
                j = image[y + dy, x + dx]
                glcm[i, j] += 1
        
        # 归一化
        glcm_sum = np.sum(glcm)
        if glcm_sum > 0:
            glcm = glcm / glcm_sum
        
        return glcm
    except Exception as e:
        print(f"计算GLCM时出错: {e}")
        return np.zeros((levels, levels), dtype=np.float32)

def calculate_texture_features_manual(image):
    """手动计算纹理特征参数"""
    features = {}
    angles = [0, 45, 90, 135]
    levels = 16
    
    # 计算4个方向的GLCM
    glcms = []
    for angle in angles:
        glcm = calculate_glcm_manual(image, distance=1, angle=angle, levels=levels)
        glcms.append(glcm)
    
    # 准备特征列表
    energy_list = []
    contrast_list = []
    correlation_list = []
    entropy_list = []
    homogeneity_list = []
    
    for idx, glcm in enumerate(glcms):
        # 能量（角二阶矩）
        energy = np.sum(glcm ** 2)
        energy_list.append(energy)
        
        # 对比度
        contrast = 0
        for i in range(levels):
            for j in range(levels):
                contrast += glcm[i, j] * ((i - j) ** 2)
        contrast_list.append(contrast)
        
        # 相关性
        # 先计算均值和标准差
        mean_i = 0
        mean_j = 0
        for i in range(levels):
            row_sum = np.sum(glcm[i, :])
            mean_i += i * row_sum
        
        for j in range(levels):
            col_sum = np.sum(glcm[:, j])
            mean_j += j * col_sum
        
        std_i = 0
        for i in range(levels):
            row_sum = np.sum(glcm[i, :])
            std_i += ((i - mean_i) ** 2) * row_sum
        std_i = np.sqrt(std_i)
        
        std_j = 0
        for j in range(levels):
            col_sum = np.sum(glcm[:, j])
            std_j += ((j - mean_j) ** 2) * col_sum
        std_j = np.sqrt(std_j)
        
        correlation = 0
        if std_i > 0 and std_j > 0:
            for i in range(levels):
                for j in range(levels):
                    correlation += glcm[i, j] * (i - mean_i) * (j - mean_j)
            correlation /= (std_i * std_j)
        correlation_list.append(correlation)
        
        # 熵
        entropy = 0
        for i in range(levels):
            for j in range(levels):
                if glcm[i, j] > 0:
                    entropy -= glcm[i, j] * np.log(glcm[i, j] + 1e-10)
        entropy_list.append(entropy)
        
        # 同质性
        homogeneity = 0
        for i in range(levels):
            for j in range(levels):
                if i != j:  # 避免除以0
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
        homogeneity_list.append(homogeneity)
    
    # 存储特征
    angle_names = ['0', '45', '90', '135']
    for i, angle in enumerate(angle_names):
        features[f'Energy_{angle}'] = energy_list[i]
        features[f'Contrast_{angle}'] = contrast_list[i]
        features[f'Correlation_{angle}'] = correlation_list[i]
        features[f'Entropy_{angle}'] = entropy_list[i]
        features[f'Homogeneity_{angle}'] = homogeneity_list[i]
    
    # 平均特征
    features['Energy_avg'] = np.mean(energy_list)
    features['Contrast_avg'] = np.mean(contrast_list)
    features['Correlation_avg'] = np.mean(correlation_list)
    features['Entropy_avg'] = np.mean(entropy_list)
    features['Homogeneity_avg'] = np.mean(homogeneity_list)
    
    return features

# ==================== 第一部分：纹理特征提取 ====================

def part1_texture_analysis():
    """第一部分：纹理特征提取和显示"""
    base_path = r'D:\cv'
    
    # 根据您提供的文件列表，只处理这些图像
    # A1-A3, A11-A31 中的存在文件
    part1_images = ['A1.png', 'A2.png', 'A3.png', 'A11.png', 'A21.png', 'A31.png']
    
    all_features = []
    
    print("=" * 100)
    print("第一部分：纹理特征提取结果")
    print("测试图像: A1-A3, A11, A21, A31")
    print("=" * 100)
    
    for img_name in part1_images:
        exists, img_path = check_image_exists(base_path, img_name)
        if not exists:
            continue
        
        print(f"\n处理图像: {img_name}")
        
        # 预处理图像
        processed_img = preprocess_image(img_path)
        if processed_img is None:
            continue
        
        # 计算纹理特征
        features = calculate_texture_features_manual(processed_img)
        features['Image'] = img_name
        all_features.append(features)
        
        # 显示简要结果
        print(f"平均能量: {features['Energy_avg']:.4f}")
        print(f"平均对比度: {features['Contrast_avg']:.4f}")
        print(f"平均相关性: {features['Correlation_avg']:.4f}")
        print(f"平均熵: {features['Entropy_avg']:.4f}")
        print(f"平均同质性: {features['Homogeneity_avg']:.4f}")
    
    if not all_features:
        print("没有成功提取任何图像的特征！")
        return None
    
    # 创建DataFrame并显示
    df = pd.DataFrame(all_features)
    
    # 重新排序列顺序，将图像名放在第一列
    cols = ['Image'] + [col for col in df.columns if col != 'Image']
    df = df[cols]
    
    print("\n" + "=" * 100)
    print("纹理特征参数详细结果表：")
    print("=" * 100)
    
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df)
    
    return df

# ==================== 第二部分：疵点检测与分割 ====================

def calculate_window_features(window, levels=16):
    """计算窗口的纹理特征"""
    # 计算GLCM（0度方向）
    glcm = calculate_glcm_manual(window, distance=1, angle=0, levels=levels)
    
    # 计算对比度
    contrast = 0
    for i in range(levels):
        for j in range(levels):
            contrast += glcm[i, j] * ((i - j) ** 2)
    
    # 计算能量
    energy = np.sum(glcm ** 2)
    
    # 计算熵
    entropy = 0
    for i in range(levels):
        for j in range(levels):
            if glcm[i, j] > 0:
                entropy -= glcm[i, j] * np.log(glcm[i, j] + 1e-10)
    
    return energy, contrast, entropy

def defect_detection(image_path):
    """基于纹理特征的疵点检测与分割"""
    try:
        # 读取并预处理图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"无法读取图像（可能格式不支持）: {image_path}")
            return None, None, None
        
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_16 = (gray // 16).astype(np.uint8)
        
        # 设置滑动窗口参数
        window_size = 32
        step_size = 16
        
        height, width = gray_16.shape
        defect_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 使用当前图像计算参考特征
        normal_features = calculate_texture_features_manual(gray_16)
        
        # 获取正常区域的参考值
        energy_ref = normal_features['Energy_avg']
        contrast_ref = normal_features['Contrast_avg']
        entropy_ref = normal_features['Entropy_avg']
        
        # 滑动窗口遍历图像
        for y in range(0, height - window_size, step_size):
            for x in range(0, width - window_size, step_size):
                # 提取窗口
                window = gray_16[y:y+window_size, x:x+window_size]
                
                # 计算窗口的纹理特征
                energy_window, contrast_window, entropy_window = calculate_window_features(window)
                
                # 判断是否为疵点区域
                energy_diff = abs(energy_window - energy_ref) / (energy_ref + 1e-10)
                contrast_diff = abs(contrast_window - contrast_ref) / (contrast_ref + 1e-10)
                entropy_diff = abs(entropy_window - entropy_ref) / (entropy_ref + 1e-10)
                
                # 组合多个特征进行判断
                defect_score = 0.4 * contrast_diff + 0.3 * energy_diff + 0.3 * entropy_diff
                
                if defect_score > 0.3:  # 阈值可以根据实际情况调整
                    defect_mask[y:y+window_size, x:x+window_size] = 255
        
        # 形态学操作，改善分割结果
        kernel = np.ones((5, 5), np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        
        # 在原始图像上标记疵点区域
        result_img = original_img.copy()
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
        
        # 创建二值化的分割结果
        segmented_defects = cv2.bitwise_and(original_img, original_img, mask=defect_mask)
        
        return result_img, defect_mask, segmented_defects
    
    except Exception as e:
        print(f"疵点检测时出错 {image_path}: {e}")
        return None, None, None

def part2_defect_detection():
    """第二部分：疵点检测与分割"""
    base_path = r'D:\cv'
    
    # 根据您提供的文件列表，只处理这些图像
    # A11-A31, A4 中的存在文件
    part2_images = ['A11.png', 'A21.png', 'A31.png', 'A4.png']
    
    print("\n" + "=" * 100)
    print("第二部分：疵点检测与分割")
    print("测试图像: A11, A21, A31, A4")
    print("=" * 100)
    
    successful_images = []
    
    for img_name in part2_images:
        exists, img_path = check_image_exists(base_path, img_name)
        if not exists:
            continue
        
        print(f"\n处理图像: {img_name}")
        
        # 执行疵点检测
        result_img, defect_mask, segmented_defects = defect_detection(img_path)
        
        if result_img is not None and defect_mask is not None:
            successful_images.append((img_name, result_img, defect_mask, segmented_defects))
    
    if not successful_images:
        print("没有成功检测到任何图像的疵点！")
        return
    
    # 创建图像显示窗口
    fig = plt.figure(figsize=(20, 12))
    
    # 显示结果
    for i, (img_name, result_img, defect_mask, segmented_defects) in enumerate(successful_images):
        # 读取原始图像用于显示
        img_path = os.path.join(base_path, img_name)
        original_img = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 显示原始图像
        plt.subplot(len(successful_images), 4, i*4 + 1)
        plt.imshow(original_rgb)
        plt.title(f'{img_name} - 原图')
        plt.axis('off')
        
        # 显示疵点掩膜
        plt.subplot(len(successful_images), 4, i*4 + 2)
        plt.imshow(defect_mask, cmap='gray')
        plt.title('疵点掩膜')
        plt.axis('off')
        
        # 显示检测结果
        plt.subplot(len(successful_images), 4, i*4 + 3)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb)
        plt.title('检测结果(红色为疵点)')
        plt.axis('off')
        
        # 显示分割出的疵点
        plt.subplot(len(successful_images), 4, i*4 + 4)
        segmented_rgb = cv2.cvtColor(segmented_defects, cv2.COLOR_BGR2RGB)
        plt.imshow(segmented_rgb)
        plt.title('分割出的疵点')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 100)
    print(f"成功处理了 {len(successful_images)} 张图像")
    print("疵点检测完成！红色轮廓标出了检测到的疵点区域。")
    print("=" * 100)

# ==================== 主程序 ====================

def main():
    print("针织物疵点检测系统")
    print("=" * 100)
    print("图像路径: D:\\cv\\")
    print("=" * 100)
    
    # 检查目录是否存在
    if not os.path.exists(r'D:\cv'):
        print("错误：目录 D:\\cv 不存在！")
        print("请创建目录或将图像文件放在正确的位置。")
        return
    
    print("开始检查图像文件...")
    
    # 执行第一部分：纹理特征提取
    print("\n开始执行第一部分：纹理特征提取")
    texture_features_df = part1_texture_analysis()
    
    # 执行第二部分：疵点检测与分割
    print("\n开始执行第二部分：疵点检测与分割")
    part2_defect_detection()
    
    print("\n" + "=" * 100)
    print("程序执行完成！")
    print("=" * 100)

if __name__ == "__main__":
    main()