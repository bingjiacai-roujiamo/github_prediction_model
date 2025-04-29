import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information

class MultiModalRegistration:
    def __init__(self, fixed_path, moving_path):
        """
        初始化配准器
        :param fixed_path: 参考图像路径（光镜染色图像）
        :param moving_path: 待配准图像路径（SHG图像）
        """
        # 读取图像并预处理
        self.fixed = self._load_and_preprocess(fixed_path, modality='histology')
        self.moving = self._load_and_preprocess(moving_path, modality='shg')
        
        # 存储中间结果
        self.registration_result = None
        self.transform_parameters = None
        
    def _load_and_preprocess(self, img_path, modality):
        """
        模态特异性预处理
        """
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像：{img_path}")
            
        # 转换为单通道
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 模态特异性处理
        if modality == 'shg':
            # SHG图像增强：CLAHE + 去噪
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            img = cv2.fastNlMeansDenoising(img, h=15, templateWindowSize=7, searchWindowSize=21)
        elif modality == 'histology':
            # 光镜图像增强：自适应阈值
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 51, 5)
        
        # 统一尺寸（以fixed为基准）
        if hasattr(self, 'fixed'):
            img = cv2.resize(img, (self.fixed.shape[1], self.fixed.shape[0]))
            
        return img.astype(np.float32) / 255.0  # 归一化

    def rigid_registration(self, optimizer_iterations=200):
        """
        刚性配准（平移+旋转）
        """
        # 将numpy数组转换为SimpleITK图像
        fixed_sitk = sitk.GetImageFromArray(self.fixed)
        moving_sitk = sitk.GetImageFromArray(self.moving)
        
        # 初始化配准方法
        registration = sitk.ImageRegistrationMethod()
        
        # 设置相似性度量（适合多模态的互信息）
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        
        # 设置优化器
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=optimizer_iterations,
            relaxationFactor=0.5
        )
        
        # 初始化变换（欧拉变换）
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, 
            moving_sitk, 
            sitk.Euler2DTransform()
        )
        registration.SetInitialTransform(initial_transform)
        
        # 执行配准
        final_transform = registration.Execute(fixed_sitk, moving_sitk)
        
        # 应用变换
        self.registration_result = sitk.Resample(
            moving_sitk, 
            fixed_sitk, 
            final_transform, 
            sitk.sitkLinear, 
            0.0, 
            moving_sitk.GetPixelID()
        )
        
        # 转换为numpy数组
        self.registration_result = sitk.GetArrayFromImage(self.registration_result)
        self.transform_parameters = final_transform.GetParameters()
        
        return self.registration_result

    def evaluate_registration(self):
        """
        配准质量评估
        """
        if self.registration_result is None:
            raise RuntimeError("需要先执行配准")
            
        # 计算评估指标
        metrics = {
            'MSE': np.mean((self.fixed - self.registration_result)**2),
            'SSIM': ssim(self.fixed, self.registration_result, data_range=1.0),
            'NMI': normalized_mutual_information(self.fixed, self.registration_result),
            'EdgeOverlap': self._calc_edge_overlap()
        }
        
        return metrics
    
    def _calc_edge_overlap(self):
        """
        计算边缘重叠率
        """
        # Canny边缘检测
        fixed_edges = cv2.Canny((self.fixed*255).astype(np.uint8), 50, 150)
        reg_edges = cv2.Canny((self.registration_result*255).astype(np.uint8), 50, 150)
        
        # 计算重叠像素
        intersection = np.logical_and(fixed_edges, reg_edges)
        union = np.logical_or(fixed_edges, reg_edges)
        
        return np.sum(intersection) / np.sum(union)

    def visualize(self):
        """
        可视化配准结果
        """
        plt.figure(figsize=(18,6))
        
        # 显示原始图像
        plt.subplot(1,3,1)
        plt.imshow(self.fixed, cmap='gray')
        plt.title('Fixed Image (Histology)')
        
        plt.subplot(1,3,2)
        plt.imshow(self.moving, cmap='gray')
        plt.title('Moving Image (SHG)')
        
        # 显示配准结果
        plt.subplot(1,3,3)
        plt.imshow(self.registration_result, cmap='gray')
        plt.title('Registered SHG')
        
        # 叠加显示边缘
        fixed_edges = cv2.Canny((self.fixed*255).astype(np.uint8), 50, 150)
        reg_edges = cv2.Canny((self.registration_result*255).astype(np.uint8), 50, 150)
        
        overlay = np.zeros((*self.fixed.shape, 3))
        overlay[fixed_edges > 0] = [1,0,0]  # 红色为固定图像边缘
        overlay[reg_edges > 0] = [0,1,0]    # 绿色为配准后边缘
        
        plt.imshow(overlay, alpha=0.4)
        plt.show()

if __name__ == "__main__":
    # 使用示例
    try:
        # 初始化配准器
        registrar = MultiModalRegistration(
            fixed_path="histology_image.tif",  # 替换为实际路径
            moving_path="shg_image.tif"        # 替换为实际路径
        )
        
        # 执行刚性配准
        registered_img = registrar.rigid_registration(optimizer_iterations=300)
        
        # 评估结果
        metrics = registrar.evaluate_registration()
        print("配准质量评估:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"SSIM: {metrics['SSIM']:.3f}")
        print(f"归一化互信息: {metrics['NMI']:.3f}")
        print(f"边缘重叠率: {metrics['EdgeOverlap']:.2%}")
        
        # 可视化
        registrar.visualize()
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

