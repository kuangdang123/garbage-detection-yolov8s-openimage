# 最简单的部署代码
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
GARBAGE_CLASSIFICATION = {
    # 可回收物
    "Bottle": {
        "category": "可回收物",
        "color": (0, 255, 0),  # 绿色
        "advice": "请清空内容物，压扁后投入可回收物垃圾桶",
        "icon": "♻️"
    },
    "Book": {
        "category": "可回收物", 
        "color": (0, 255, 0),
        "advice": "保持干燥整洁，投入可回收物垃圾桶",
        "icon": "♻️"
    },
    
    # 有害垃圾
    "Mobile phone": {
        "category": "有害垃圾",
        "color": (255, 0, 0),  # 红色
        "advice": "含有重金属，请投入有害垃圾回收箱或专门回收点",
        "icon": "☣️"
    },
    
    # 厨余垃圾  
    "Banana": {
        "category": "厨余垃圾",
        "color": (255, 165, 0),  # 橙色
        "advice": "请投入厨余垃圾桶，可用于堆肥",
        "icon": "🍌"
    },
    "Apple": {
        "category": "厨余垃圾",
        "color": (255, 165, 0),
        "advice": "果核可降解，请投入厨余垃圾桶",
        "icon": "🍎"
    },
    "Orange": {
        "category": "厨余垃圾", 
        "color": (255, 165, 0),
        "advice": "果皮易腐烂，请投入厨余垃圾桶",
        "icon": "🍊"
    },
    
    # 其他垃圾
    "Plastic bag": {
        "category": "其他垃圾",
        "color": (128, 128, 128),  # 灰色
        "advice": "污染的塑料袋属于其他垃圾，请投入其他垃圾桶",
        "icon": "🛍️"
    },
    "Toilet paper": {
        "category": "其他垃圾",
        "color": (128, 128, 128),
        "advice": "使用过的卫生纸属于其他垃圾，请投入其他垃圾桶", 
        "icon": "🧻"
    },
    "Coffee cup": {
        "category": "其他垃圾",
        "color": (128, 128, 128),
        "advice": "一次性咖啡杯通常属于其他垃圾，请投入其他垃圾桶",
        "icon": "☕"
    }
}

class GarbageDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.category_colors = {
            "可回收物": (0, 255, 0),      # 绿色
            "有害垃圾": (255, 0, 0),      # 红色  
            "厨余垃圾": (255, 165, 0),    # 橙色
            "其他垃圾": (128, 128, 128)   # 灰色
        }

    def detect(self, image_input):
        """
        检测垃圾
        image_input: 可以是文件路径、PIL图像、numpy数组等
        """
        # 完全不需要手动预处理！
        results = self.model(image_input)
        return self._parse_results(results)
    
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
                    "category": "未知分类",
                    "color": (0, 0, 255),  # 蓝色
                    "advice": "请查询当地垃圾分类标准",
                    "icon": "❓"
                })
                
                detection = {
                    'id': i,
                    'class': class_name,
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
        img_copy = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            color = detection['color']
            category = detection['category']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            
            # 绘制类别标签背景
            label = f"{detection['icon']} {class_name} {category} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 绘制文本
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img_copy
    
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