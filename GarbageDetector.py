# 最简单的部署代码
from ultralytics import YOLO

class GarbageDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, image_input):
        """
        检测垃圾
        image_input: 可以是文件路径、PIL图像、numpy数组等
        """
        # 完全不需要手动预处理！
        results = self.model(image_input)
        return self._parse_results(results)
    
    def _parse_results(self, results):
        """解析检测结果"""
        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'class': self.model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        return detections

# 使用示例
detector = GarbageDetector('garbage_detection_precise/stage4_full_finetune/weights/best.pt')

# 检测新图像
results = detector.detect('new_garbage_photo.jpg')
for detection in results:
    print(f"找到 {detection['class']}, 置信度: {detection['confidence']:.2f}")