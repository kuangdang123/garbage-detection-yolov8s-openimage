# 最简单的部署代码
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from config import GARBAGE_CLASSIFICATION

class GarbageDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.category_colors = {
            "可回收物": (0, 255, 0),      # 绿色
            "有害垃圾": (255, 0, 0),      # 红色  
            "厨余垃圾": (255, 165, 0),    # 橙色
            "其他垃圾": (128, 128, 128)   # 灰色
        }

    def detect(self, image_input, confidence_threshold=0.5):
        """
        检测垃圾
        image_input: 可以是文件路径、PIL图像、numpy数组等
        """
        # 完全不需要手动预处理！
        results = self.model(image_input, conf=confidence_threshold)
        return self._parse_results(results, image_input)
    
    def _parse_results(self, results, original_image):
        """
        解析检测结果并添加垃圾分类信息
        results: 检测结果
        original_image: 原始图像
        """
        detections = []
        
        for result in results:
            # 获取原始图像（用于绘制边界框）
            if isinstance(original_image, str):
                img = cv2.imread(original_image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(original_image, Image.Image):
                img = np.array(original_image)
            else:
                img = original_image.copy()
            
            # 为每个检测结果添加详细信息
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # 获取垃圾分类信息
                garbage_info = GARBAGE_CLASSIFICATION.get(class_name, {
                    "name": "未知物品",
                    "category": "未知分类",
                    "color": (0, 0, 255),  # 蓝色
                    "advice": "请查询当地垃圾分类标准",
                    "icon": "❓"
                })
                
                detection = {
                    'id': i,
                    'class': garbage_info['name'],
                    'confidence': confidence,
                    'bbox': bbox,
                    'category': garbage_info['category'],
                    'color': garbage_info['color'],
                    'advice': garbage_info['advice'], 
                    'icon': garbage_info['icon'],
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # 边界框面积
                }
                detections.append(detection)
            
            # 在图像上绘制检测结果
            annotated_image = self._draw_detections(img, detections)
            
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'total_count': len(detections),
            'category_stats': self._get_category_statistics(detections)
        }
    
    def _draw_detections(self, image, detections):
        """
        在图像上绘制检测结果和分类信息
        image: 原始图像
        detections: 检测结果
        """
        # 将OpenCV图像转换为PIL图像以便绘制中文
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image.copy()
        
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        try:
            # 尝试几种常见的中文字体
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',  # Windows
                'C:/Windows/Fonts/msyh.ttc',    # Windows
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 20)
                        break
                    except:
                        continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            color = detection['color']
            category = detection['category']
            class_name = detection['class']
            confidence = detection['confidence']
            icon = detection['icon']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 准备标签文本（使用图标和中文）
            label_text = f"{class_name} {category} {confidence:.2f}"
            
            # 计算文本大小
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label_text) * 10
                text_height = 20
            
            # 绘制标签背景
            draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], 
                        fill=color)
            
            # 绘制文本
            try:
                draw.text((x1 + 5, y1 - text_height - 5), label_text, 
                        fill=(255, 255, 255), font=font)
            except:
                # 如果中文绘制失败，回退到英文
                label_text_en = f"{icon} {class_name} {confidence:.2f}"
                draw.text((x1 + 5, y1 - text_height - 5), label_text_en, 
                        fill=(255, 255, 255), font=font)
        
        # 转换回numpy数组
        return np.array(img_pil)
    
    def _get_category_statistics(self, detections):
        """
        统计各类垃圾的数量
        detections: 检测结果
        """
        stats = {
            "可回收物": {"count": 0, "items": []},
            "有害垃圾": {"count": 0, "items": []},
            "厨余垃圾": {"count": 0, "items": []},
            "其他垃圾": {"count": 0, "items": []},
            "未知分类": {"count": 0, "items": []}
        }
        
        for detection in detections:
            category = detection['category']
            if category in stats:
                stats[category]["count"] += 1
                stats[category]["items"].append({
                    "class": detection['class'],
                    "confidence": detection['confidence']
                })
        
        return stats