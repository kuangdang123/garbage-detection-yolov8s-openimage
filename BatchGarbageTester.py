import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path
import cv2
from PIL import Image
import yaml

class BatchGarbageTester:
    def __init__(self, model_path, test_images_dir, output_dir="batch_test_results"):
        """
        批量垃圾检测测试类
    
        model_path: 模型权重文件路径
        test_images_dir: 测试图片目录
        output_dir: 输出结果目录
        """
        self.model = YOLO(model_path)
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "detection_results").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)

        # 创建YOLO格式标注目录
        (self.output_dir / "yolo_labels").mkdir(exist_ok=True)
        (self.output_dir / "yolo_format").mkdir(exist_ok=True)
        
        # 创建评估结果目录
        (self.output_dir / "evaluation").mkdir(exist_ok=True)
        (self.output_dir / "evaluation" / "plots").mkdir(exist_ok=True)
        
        # 垃圾分类映射
        self.garbage_categories = {
            "Bottle": "可回收物",
            "Book": "可回收物",
            "Mobile phone": "有害垃圾", 
            "Banana": "厨余垃圾",
            "Apple": "厨余垃圾",
            "Orange": "厨余垃圾",
            "Plastic bag": "其他垃圾",
            "Toilet paper": "其他垃圾",
            "Coffee cup": "其他垃圾"
        }
        
        # 存储测试结果
        self.results = []
        self.detection_stats = {
            "total_images": 0,
            "total_detections": 0,
            "category_counts": {},
            "confidence_stats": [],
            "inference_times": []
        }

        # 存储类别ID映射
        self.class_to_id = {}
        self._setup_class_mapping()

    def _setup_class_mapping(self):
        """设置类别名称到ID的映射"""
        unique_classes = set(self.garbage_categories.keys())
        self.class_to_id = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        
    def run_batch_test(self, confidence_threshold=0.5, iou_threshold=0.5, max_images=None):
        """
        运行批量测试
        
        confidence_threshold: 置信度阈值
        iou_threshold: NMS IoU阈值
        max_images: 最大测试图片数量（None表示测试所有）
        """
        print("🚀 开始批量测试...")
        
        # 获取测试图片
        image_paths = list(self.test_images_dir.glob("*.jpg")) + \
                     list(self.test_images_dir.glob("*.png")) + \
                     list(self.test_images_dir.glob("*.jpeg"))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        self.detection_stats["total_images"] = len(image_paths)
        
        # 进度条
        progress_bar = tqdm(image_paths, desc="测试进度")
        
        for image_path in progress_bar:
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行检测
                results = self.model.predict(
                    source=str(image_path),
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    max_det=100,
                    verbose=False
                )
                
                # 记录推理时间
                inference_time = time.time() - start_time
                self.detection_stats["inference_times"].append(inference_time)
                
                # 处理检测结果
                image_results = self._process_single_image_results(
                    results, str(image_path), inference_time
                )
                
                self.results.append(image_results)
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    '已检测': len(self.results),
                    '平均时间': f"{np.mean(self.detection_stats['inference_times']):.3f}s"
                })
                
            except Exception as e:
                print(f"❌ 处理图片 {image_path} 时出错: {e}")
                continue
        
        print("✅ 批量测试完成!")
        
        # 生成报告
        self._generate_test_report(confidence_threshold, iou_threshold)
        
        # 在生成报告后添加YOLO格式导出
        self._generate_yolo_format_export()

        return self.results
    
    def _process_single_image_results(self, results, image_path, inference_time):
        """处理单张图片的检测结果"""
        image_results = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "inference_time": inference_time,
            "detections": [],
            "detection_count": 0,
            "categories_found": set(),
            "image_size": None
        }
        
        for result in results:
            # 获取图片尺寸
            if result.orig_shape is not None:
                image_results["image_size"] = result.orig_shape  # (height, width)

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    # 获取垃圾分类
                    garbage_category = self.garbage_categories.get(class_name, "未知分类")
                    
                    detection_info = {
                        "class_name": class_name,
                        "category": garbage_category,
                        "confidence": confidence,
                        "bbox": bbox,
                        "bbox_normalized": None,
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }

                    # 计算归一化边界框 (YOLO格式)
                    if image_results["image_size"] is not None:
                        img_h, img_w = image_results["image_size"]
                        x_center = (bbox[0] + bbox[2]) / 2 / img_w
                        y_center = (bbox[1] + bbox[3]) / 2 / img_h
                        width = (bbox[2] - bbox[0]) / img_w
                        height = (bbox[3] - bbox[1]) / img_h
                        detection_info["bbox_normalized"] = [x_center, y_center, width, height]
                    
                    image_results["detections"].append(detection_info)
                    image_results["categories_found"].add(garbage_category)
                    
                    # 更新统计信息
                    self.detection_stats["total_detections"] += 1
                    self.detection_stats["category_counts"][garbage_category] = \
                        self.detection_stats["category_counts"].get(garbage_category, 0) + 1
                    self.detection_stats["confidence_stats"].append(confidence)
        
        image_results["detection_count"] = len(image_results["detections"])
        image_results["categories_found"] = list(image_results["categories_found"])
        
        return image_results
    
    def _generate_test_report(self, confidence_threshold, iou_threshold):
        """生成测试报告"""
        print("📊 生成测试报告...")
        
        # 1. 保存原始结果
        self._save_raw_results()
        
        # 2. 生成统计报告
        self._generate_statistical_report(confidence_threshold, iou_threshold)
        
        # 3. 生成可视化图表
        self._generate_visualizations()
        
        # 4. 生成YOLO格式的评估指标
        self._generate_yolo_metrics()
    
    def _save_raw_results(self):
        """保存原始检测结果"""
        # 保存为JSON
        with open(self.output_dir / "raw_detection_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV
        csv_data = []
        for result in self.results:
            for detection in result["detections"]:
                csv_data.append({
                    "image_path": result["image_path"],
                    "image_name": result["image_name"],
                    "class_name": detection["class_name"],
                    "category": detection["category"],
                    "confidence": detection["confidence"],
                    "inference_time": result["inference_time"]
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.output_dir / "detection_results.csv", index=False, encoding="utf-8")
    
    def _generate_statistical_report(self, confidence_threshold, iou_threshold):
        """生成统计报告"""
        stats = self.detection_stats
        
        # 基本统计
        report = {
            "测试配置": {
                "模型路径": str(self.model.ckpt_path),
                "测试图片数量": stats["total_images"],
                "置信度阈值": confidence_threshold,
                "IoU阈值": iou_threshold,
                "总检测数量": stats["total_detections"]
            },
            "性能统计": {
                "平均推理时间": f"{np.mean(stats['inference_times']):.4f}秒",
                "最快推理时间": f"{np.min(stats['inference_times']):.4f}秒",
                "最慢推理时间": f"{np.max(stats['inference_times']):.4f}秒",
                "FPS": f"{1/np.mean(stats['inference_times']):.2f}",
                "总测试时间": f"{np.sum(stats['inference_times']):.2f}秒"
            },
            "检测统计": {
                "平均每张图片检测数": f"{stats['total_detections'] / stats['total_images']:.2f}",
                "检测率": f"{(sum(1 for r in self.results if r['detection_count'] > 0) / stats['total_images']) * 100:.2f}%",
                "平均置信度": f"{np.mean(stats['confidence_stats']):.4f}",
                "置信度标准差": f"{np.std(stats['confidence_stats']):.4f}"
            },
            "分类统计": stats["category_counts"]
        }
        
        # 保存统计报告
        with open(self.output_dir / "statistical_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
    
    def _generate_markdown_report(self, report):
        """生成Markdown格式的报告"""
        md_content = f"""# 垃圾检测批量测试报告

## 测试配置
- **模型**: {report['测试配置']['模型路径']}
- **测试图片数量**: {report['测试配置']['测试图片数量']}
- **置信度阈值**: {report['测试配置']['置信度阈值']}
- **IoU阈值**: {report['测试配置']['IoU阈值']}
- **总检测数量**: {report['测试配置']['总检测数量']}

## 性能统计
- **平均推理时间**: {report['性能统计']['平均推理时间']}
- **FPS**: {report['性能统计']['FPS']}
- **总测试时间**: {report['性能统计']['总测试时间']}

## 检测统计
- **平均每张图片检测数**: {report['检测统计']['平均每张图片检测数']}
- **检测率**: {report['检测统计']['检测率']}
- **平均置信度**: {report['检测统计']['平均置信度']}

## 垃圾分类统计
"""
        
        for category, count in report['分类统计'].items():
            percentage = (count / report['测试配置']['总检测数量']) * 100
            md_content += f"- **{category}**: {count} 个 ({percentage:.1f}%)\\n"
        
        # 保存Markdown报告
        with open(self.output_dir / "test_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        print("📈 生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 垃圾分类分布饼图
        self._plot_category_distribution()
        
        # 2. 检测置信度分布直方图
        self._plot_confidence_distribution()
        
        # 3. 推理时间分布图
        self._plot_inference_time_distribution()
        
        # 4. 每张图片检测数量分布
        self._plot_detections_per_image()
        
        # 5. 各类别检测数量柱状图
        self._plot_category_bar_chart()
    
    def _plot_category_distribution(self):
        """绘制垃圾分类分布饼图"""
        category_counts = self.detection_stats["category_counts"]
        
        if not category_counts:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E', '#2196F3']
        
        wedges, texts, autotexts = ax.pie(
            category_counts.values(),
            labels=category_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(category_counts)]
        )
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('垃圾分类分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "category_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self):
        """绘制置信度分布直方图"""
        if not self.detection_stats["confidence_stats"]:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.detection_stats["confidence_stats"], bins=20, 
               alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_xlabel('置信度')
        ax.set_ylabel('频率')
        ax.set_title('检测置信度分布')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_conf = np.mean(self.detection_stats["confidence_stats"])
        ax.axvline(mean_conf, color='red', linestyle='--', 
                  label=f'平均置信度: {mean_conf:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "confidence_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_time_distribution(self):
        """绘制推理时间分布图"""
        if not self.detection_stats["inference_times"]:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.detection_stats["inference_times"], bins=20, 
               alpha=0.7, color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('推理时间 (秒)')
        ax.set_ylabel('频率')
        ax.set_title('推理时间分布')
        ax.grid(True, alpha=0.3)
        
        mean_time = np.mean(self.detection_stats["inference_times"])
        ax.axvline(mean_time, color='red', linestyle='--', 
                  label=f'平均时间: {mean_time:.3f}s')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "inference_time_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detections_per_image(self):
        """绘制每张图片检测数量分布"""
        detections_per_image = [r["detection_count"] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(detections_per_image, bins=20, alpha=0.7, 
               color='orange', edgecolor='black')
        
        ax.set_xlabel('每张图片检测数量')
        ax.set_ylabel('图片数量')
        ax.set_title('每张图片检测数量分布')
        ax.grid(True, alpha=0.3)
        
        mean_detections = np.mean(detections_per_image)
        ax.axvline(mean_detections, color='red', linestyle='--', 
                  label=f'平均检测数: {mean_detections:.2f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "detections_per_image.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_bar_chart(self):
        """绘制各类别检测数量柱状图"""
        category_counts = self.detection_stats["category_counts"]
        
        if not category_counts:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        bars = ax.bar(categories, counts, color=['#4CAF50', '#F44336', '#FF9800', '#9E9E9E'])
        
        ax.set_xlabel('垃圾类别')
        ax.set_ylabel('检测数量')
        ax.set_title('各类别检测数量')
        ax.tick_params(axis='x', rotation=45)
        
        # 在柱子上添加数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "category_bar_chart.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_yolo_metrics(self):
        """生成YOLO格式的评估指标"""
        # 如果有标注文件，可以使用YOLO的val方法
        # 这里我们生成一些基于检测结果的衍生指标
        
        metrics = {
            "检测覆盖率": self._calculate_detection_coverage(),
            "类别平衡度": self._calculate_category_balance(),
            "检测稳定性": self._calculate_detection_stability(),
            "性能指标": {
                "平均FPS": 1 / np.mean(self.detection_stats["inference_times"]),
                "吞吐量": len(self.results) / np.sum(self.detection_stats["inference_times"]),
                "检测密度": self.detection_stats["total_detections"] / self.detection_stats["total_images"]
            }
        }
        
        with open(self.output_dir / "metrics" / "yolo_style_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    def _calculate_detection_coverage(self):
        """计算检测覆盖率"""
        images_with_detections = sum(1 for r in self.results if r["detection_count"] > 0)
        return images_with_detections / self.detection_stats["total_images"]
    
    def _calculate_category_balance(self):
        """计算类别平衡度"""
        category_counts = list(self.detection_stats["category_counts"].values())
        if not category_counts:
            return 0
        return min(category_counts) / max(category_counts)
    
    def _calculate_detection_stability(self):
        """计算检测稳定性（检测数量的变异系数）"""
        detection_counts = [r["detection_count"] for r in self.results]
        if not detection_counts:
            return 0
        return np.std(detection_counts) / np.mean(detection_counts)
    
    def get_summary(self):
        """获取测试摘要"""
        return {
            "total_images": self.detection_stats["total_images"],
            "total_detections": self.detection_stats["total_detections"],
            "avg_inference_time": np.mean(self.detection_stats["inference_times"]),
            "avg_confidence": np.mean(self.detection_stats["confidence_stats"]),
            "category_breakdown": self.detection_stats["category_counts"]
        }
    
    def _generate_yolo_format_export(self):
        """生成YOLO格式的标注文件和数据集配置"""
        print("📝 生成YOLO格式标注文件...")
        
        # 1. 生成YOLO格式的标签文件
        self._generate_yolo_labels()
        
        # 2. 生成数据集YAML配置文件
        self._generate_dataset_yaml()
        
        # 3. 创建数据集的目录结构
        self._create_dataset_structure()
    
    def _generate_yolo_labels(self):
        """生成YOLO格式的标签文件"""
        labels_dir = self.output_dir / "yolo_labels"
        
        for result in self.results:
            if result["detections"] and result["image_size"] is not None:
                # 生成对应的标签文件名
                label_filename = Path(result["image_name"]).stem + ".txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, "w", encoding="utf-8") as f:
                    for detection in result["detections"]:
                        if detection["bbox_normalized"] is not None:
                            x_center, y_center, width, height = detection["bbox_normalized"]
                            class_id = self.class_to_id.get(detection["class_name"], 0)
                            
                            # YOLO格式: class_id x_center y_center width height
                            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                            f.write(line)
    
    def _generate_dataset_yaml(self):
        """生成数据集YAML配置文件"""
        dataset_config = {
            'path': str(self.output_dir / "yolo_format"),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {idx: name for name, idx in self.class_to_id.items()}
        }
        
        yaml_path = self.output_dir / "yolo_format" / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        # 同时保存为JSON格式便于其他工具使用
        dataset_info = {
            "dataset_info": {
                "name": "garbage_detection_dataset",
                "description": "自动生成的垃圾分类检测数据集",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_images": len(self.results),
                "total_detections": self.detection_stats["total_detections"],
                "classes": list(self.class_to_id.keys()),
                "class_mapping": self.class_to_id,
                "garbage_categories": self.garbage_categories
            },
            "paths": {
                "images_dir": "images",
                "labels_dir": "labels",
                "original_images": str(self.test_images_dir)
            }
        }
        
        json_path = self.output_dir / "yolo_format" / "dataset_info.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    def _create_dataset_structure(self):
        """创建标准的数据集目录结构"""
        yolo_dir = self.output_dir / "yolo_format"
        
        # 创建标准目录结构
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        
        # 复制标签文件到labels/train目录（默认全部作为训练集）
        labels_dir = self.output_dir / "yolo_labels"
        target_labels_dir = yolo_dir / "labels" / "train"
        
        for label_file in labels_dir.glob("*.txt"):
            target_path = target_labels_dir / label_file.name
            if target_path.exists():
                target_path.unlink()
            label_file.rename(target_path)
        
        # 创建图片文件的符号链接（如果可能）或记录图片路径
        self._create_image_links_or_list(yolo_dir)
    
    def _create_image_links_or_list(self, yolo_dir):
        """创建图片链接或生成图片路径列表"""
        # 方法1: 尝试创建符号链接
        try:
            target_images_dir = yolo_dir / "images" / "train"
            for result in self.results:
                image_path = Path(result["image_path"])
                if image_path.exists():
                    link_path = target_images_dir / result["image_name"]
                    if not link_path.exists():
                        # 在Windows上可能需要管理员权限，所以尝试复制
                        import shutil
                        shutil.copy2(image_path, link_path)
            print("✅ 图片文件已复制到YOLO格式目录")
        except Exception as e:
            print(f"⚠️ 无法创建图片链接/复制文件: {e}")
            # 方法2: 生成图片路径列表文件
            self._generate_image_list_file(yolo_dir)
    
    def _generate_image_list_file(self, yolo_dir):
        """生成包含图片路径的列表文件"""
        # 生成训练集图片路径列表
        train_list_path = yolo_dir / "train.txt"
        with open(train_list_path, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(str(Path(result["image_path"]).absolute()) + '\n')
        
        print("✅ 已生成图片路径列表文件 (train.txt)")

    # =======================对标签测试=====================
    def evaluate_with_labels(self, labels_dir, data_yaml=None, batch_size=16, imgsz=640, 
                           conf_threshold=0.001, iou_threshold=0.6, save_json=False):
        """
        使用真实标签进行评估，计算各种指标
        
        labels_dir: 真实标签目录
        data_yaml: 数据集配置文件路径
        batch_size: 批处理大小
        imgsz: 图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_json: 是否保存JSON格式结果
        """
        print("📊 开始模型评估（使用真实标签）...")
        
        # 验证标签目录存在
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            print(f"❌ 标签目录不存在: {labels_dir}")
            return None
        
        # 如果未提供data_yaml，尝试自动创建
        if data_yaml is None:
            data_yaml = self._create_evaluation_data_yaml(labels_dir)
        
        # 执行评估
        try:
            results = self.model.val(
                data=data_yaml,
                batch=batch_size,
                imgsz=imgsz,
                conf=conf_threshold,
                iou=iou_threshold,
                save_json=save_json,
                project=str(self.output_dir / "evaluation"),
                name="val_results",
                exist_ok=True
            )
            
            # 处理评估结果
            self._process_evaluation_results(results)
            
            print("✅ 模型评估完成!")
            return results
            
        except Exception as e:
            print(f"❌ 评估过程中出错: {e}")
            return None
    
    def _create_evaluation_data_yaml(self, labels_dir):
        """为评估创建临时的数据集配置文件"""
        eval_data = {
            'path': str(Path(labels_dir).parent),  # 假设图片和标签在同一个父目录下
            'train': None,
            'val': str(self.test_images_dir),
            'test': None,
            'nc': len(self.class_to_id),
            'names': {v: k for k, v in self.class_to_id.items()}
        }
        
        # 保存临时YAML文件
        temp_yaml = self.output_dir / "evaluation" / "temp_data.yaml"
        with open(temp_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(eval_data, f, default_flow_style=False, allow_unicode=True)
        
        return str(temp_yaml)
    
    def _process_evaluation_results(self, results):
        """处理评估结果并生成报告和图表"""
        if results is None:
            return
        
        # 保存评估指标
        metrics = {
            "precision": results.box.map50,  # 使用mAP50作为precision的近似
            "recall": getattr(results, 'recall', None),
            "map50": results.box.map50,
            "map50_95": results.box.map,
            "f1_score": self._calculate_f1_score(results),
            "losses": {
                "box_loss": getattr(results, 'box_loss', None),
                "cls_loss": getattr(results, 'cls_loss', None),
                "dfl_loss": getattr(results, 'dfl_loss', None),
            },
            "speed": {
                "preprocess": getattr(results, 'speed', {}).get('preprocess', None),
                "inference": getattr(results, 'speed', {}).get('inference', None),
                "postprocess": getattr(results, 'speed', {}).get('postprocess', None),
            },
            "per_class_metrics": self._extract_per_class_metrics(results)
        }
        
        # 保存指标
        with open(self.output_dir / "evaluation" / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 生成评估报告
        self._generate_evaluation_report(metrics)
        
        # 生成评估图表
        self._generate_evaluation_plots(results, metrics)
    
    def _calculate_f1_score(self, results):
        """计算F1分数"""
        precision = results.box.map50  # 近似值
        recall = getattr(results, 'recall', None)
        
        if precision and recall and (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return None
    
    def _extract_per_class_metrics(self, results):
        """提取每个类别的指标"""
        per_class_metrics = {}
        
        # 尝试从结果中获取每个类别的AP
        if hasattr(results, 'results_dict') and 'results_per_class' in results.results_dict:
            for class_name, class_metrics in results.results_dict['results_per_class'].items():
                per_class_metrics[class_name] = {
                    "precision": class_metrics.get('precision', None),
                    "recall": class_metrics.get('recall', None),
                    "ap50": class_metrics.get('AP50', None),
                    "ap50_95": class_metrics.get('AP', None)
                }
        
        return per_class_metrics
    
    def _generate_evaluation_report(self, metrics):
        """生成评估报告"""
        report = {
            "评估摘要": {
                "模型路径": str(self.model.ckpt_path),
                "测试图片数量": self.detection_stats["total_images"],
                "评估时间": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "性能指标": {
                "mAP@0.5": f"{metrics['map50']:.4f}",
                "mAP@0.5:0.95": f"{metrics['map50_95']:.4f}",
                "精确率": f"{metrics['precision']:.4f}" if metrics['precision'] else "N/A",
                "召回率": f"{metrics['recall']:.4f}" if metrics['recall'] else "N/A",
                "F1分数": f"{metrics['f1_score']:.4f}" if metrics['f1_score'] else "N/A"
            },
            "损失值": metrics['losses'],
            "推理速度": metrics['speed']
        }
        
        # 保存JSON报告
        with open(self.output_dir / "evaluation" / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成Markdown报告
        self._generate_evaluation_markdown_report(report, metrics)
    
    def _generate_evaluation_markdown_report(self, report, metrics):
        """生成Markdown格式的评估报告"""
        md_content = f"""# 模型评估报告

## 评估配置
- **模型**: {report['评估摘要']['模型路径']}
- **测试图片数量**: {report['评估摘要']['测试图片数量']}
- **评估时间**: {report['评估摘要']['评估时间']}

## 性能指标
| 指标 | 值 |
|------|-----|
| mAP@0.5 | {report['性能指标']['mAP@0.5']} |
| mAP@0.5:0.95 | {report['性能指标']['mAP@0.5:0.95']} |
| 精确率 | {report['性能指标']['精确率']} |
| 召回率 | {report['性能指标']['召回率']} |
| F1分数 | {report['性能指标']['F1分数']} |

## 损失值
"""
        
        for loss_name, loss_value in report['损失值'].items():
            if loss_value is not None:
                md_content += f"- **{loss_name}**: {loss_value:.4f}\\n"
            else:
                md_content += f"- **{loss_name}**: N/A\\n"
        
        md_content += "\n## 推理速度\\n"
        for speed_name, speed_value in report['推理速度'].items():
            if speed_value is not None:
                md_content += f"- **{speed_name}**: {speed_value:.4f} ms/image\\n"
            else:
                md_content += f"- **{speed_name}**: N/A\\n"
        
        # 添加每个类别的指标
        if metrics['per_class_metrics']:
            md_content += "\n## 每个类别性能\\n"
            md_content += "| 类别 | AP@0.5 | AP@0.5:0.95 |\\n"
            md_content += "|------|--------|-------------|\\n"
            
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                ap50 = class_metrics.get('ap50', 'N/A')
                ap50_95 = class_metrics.get('ap50_95', 'N/A')
                
                if ap50 != 'N/A':
                    ap50 = f"{ap50:.4f}"
                if ap50_95 != 'N/A':
                    ap50_95 = f"{ap50_95:.4f}"
                
                md_content += f"| {class_name} | {ap50} | {ap50_95} |\\n"
        
        # 保存Markdown报告
        with open(self.output_dir / "evaluation" / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)
    
    def _generate_evaluation_plots(self, results, metrics):
        """生成评估图表"""
        print("📈 生成评估图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 绘制PR曲线
        self._plot_pr_curve(results)
        
        # 2. 绘制F1曲线
        self._plot_f1_curve(results)
        
        # 3. 绘制混淆矩阵
        self._plot_confusion_matrix(results)
        
        # 4. 绘制每个类别的AP
        self._plot_per_class_ap(results)
        
        # 5. 绘制损失曲线（如果有训练历史）
        self._plot_loss_curves_if_available()
        
        # 6. 绘制指标雷达图
        self._plot_metrics_radar(metrics)
    
    def _plot_pr_curve(self, results):
        """绘制PR曲线"""
        try:
            # 尝试从结果中获取PR曲线数据
            if hasattr(results, 'pr_curve'):
                pr_curve_data = results.pr_curve
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(pr_curve_data[0], pr_curve_data[1], linewidth=2, color='blue')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                # 添加AP值
                ap = getattr(results, 'box', {}).get('map50', 0)
                ax.text(0.6, 0.1, f'mAP@0.5: {ap:.3f}', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "pr_curve.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制PR曲线: {e}")
    
    def _plot_f1_curve(self, results):
        """绘制F1曲线"""
        try:
            # 尝试从结果中获取F1曲线数据
            if hasattr(results, 'f1_curve'):
                f1_curve_data = results.f1_curve
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(f1_curve_data[0], f1_curve_data[1], linewidth=2, color='green')
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel('F1 Score')
                ax.set_title('F1-Confidence Curve')
                ax.grid(True, alpha=0.3)
                
                # 找到最佳F1分数和对应的置信度阈值
                if len(f1_curve_data[1]) > 0:
                    best_f1_idx = np.argmax(f1_curve_data[1])
                    best_f1 = f1_curve_data[1][best_f1_idx]
                    best_conf = f1_curve_data[0][best_f1_idx]
                    
                    ax.axvline(best_conf, color='red', linestyle='--', 
                              label=f'最佳阈值: {best_conf:.3f} (F1={best_f1:.3f})')
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "f1_curve.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制F1曲线: {e}")
    
    def _plot_confusion_matrix(self, results):
        """绘制混淆矩阵"""
        try:
            if hasattr(results, 'confusion_matrix'):
                cm = results.confusion_matrix
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # 归一化混淆矩阵
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # 获取类别名称
                class_names = list(self.class_to_id.keys())
                if len(class_names) < cm.shape[0]:
                    class_names = [f"Class {i}" for i in range(cm.shape[0])]
                
                # 绘制热力图
                im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # 设置刻度
                ax.set(xticks=np.arange(cm_normalized.shape[1]),
                      yticks=np.arange(cm_normalized.shape[0]),
                      xticklabels=class_names,
                      yticklabels=class_names,
                      title="混淆矩阵 (归一化)",
                      ylabel="真实标签",
                      xlabel="预测标签")
                
                # 旋转x轴标签
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # 在格子中显示数值
                thresh = cm_normalized.max() / 2.
                for i in range(cm_normalized.shape[0]):
                    for j in range(cm_normalized.shape[1]):
                        ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                               ha="center", va="center",
                               color="white" if cm_normalized[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "confusion_matrix.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制混淆矩阵: {e}")
    
    def _plot_per_class_ap(self, results):
        """绘制每个类别的AP"""
        try:
            if metrics := getattr(results, 'results_dict', {}).get('results_per_class', {}):
                class_names = []
                ap50_values = []
                ap50_95_values = []
                
                for class_name, class_metrics in metrics.items():
                    class_names.append(class_name)
                    ap50_values.append(class_metrics.get('AP50', 0))
                    ap50_95_values.append(class_metrics.get('AP', 0))
                
                x = np.arange(len(class_names))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(12, 6))
                rects1 = ax.bar(x - width/2, ap50_values, width, label='AP@0.5', alpha=0.8)
                rects2 = ax.bar(x + width/2, ap50_95_values, width, label='AP@0.5:0.95', alpha=0.8)
                
                ax.set_xlabel('类别')
                ax.set_ylabel('AP值')
                ax.set_title('每个类别的平均精度(AP)')
                ax.set_xticks(x)
                ax.set_xticklabels(class_names, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.3f}',
                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8)
                
                autolabel(rects1)
                autolabel(rects2)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "per_class_ap.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制每个类别AP图: {e}")
    
    def _plot_loss_curves_if_available(self):
        """如果可用，绘制损失曲线"""
        try:
            # 尝试从模型路径获取训练历史
            model_dir = Path(self.model.ckpt_path).parent if self.model.ckpt_path else None
            if model_dir and (model_dir / "results.csv").exists():
                results_csv = pd.read_csv(model_dir / "results.csv")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # 绘制训练损失
                if 'train/box_loss' in results_csv.columns:
                    axes[0].plot(results_csv['train/box_loss'], label='Box Loss')
                    axes[0].plot(results_csv['train/cls_loss'], label='Cls Loss')
                    axes[0].plot(results_csv['train/dfl_loss'], label='DFL Loss')
                    axes[0].set_title('训练损失')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                
                # 绘制验证损失
                if 'val/box_loss' in results_csv.columns:
                    axes[1].plot(results_csv['val/box_loss'], label='Box Loss')
                    axes[1].plot(results_csv['val/cls_loss'], label='Cls Loss')
                    axes[1].plot(results_csv['val/dfl_loss'], label='DFL Loss')
                    axes[1].set_title('验证损失')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                # 绘制mAP曲线
                if 'metrics/mAP50(B)' in results_csv.columns:
                    axes[2].plot(results_csv['metrics/mAP50(B)'], label='mAP@0.5')
                    axes[2].plot(results_csv['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
                    axes[2].set_title('mAP指标')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
                
                # 绘制精确率和召回率
                if 'metrics/precision(B)' in results_csv.columns:
                    axes[3].plot(results_csv['metrics/precision(B)'], label='Precision')
                    axes[3].plot(results_csv['metrics/recall(B)'], label='Recall')
                    axes[3].set_title('精确率和召回率')
                    axes[3].legend()
                    axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "loss_curves.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制损失曲线: {e}")
    
    def _plot_metrics_radar(self, metrics):
        """绘制指标雷达图"""
        try:
            # 选择要展示的指标
            radar_metrics = {
                'mAP@0.5': metrics.get('map50', 0),
                'mAP@0.5:0.95': metrics.get('map50_95', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0) or 0,
                'F1 Score': metrics.get('f1_score', 0) or 0
            }
            
            # 过滤掉值为None的指标
            radar_metrics = {k: v for k, v in radar_metrics.items() if v is not None}
            
            if len(radar_metrics) >= 3:
                categories = list(radar_metrics.keys())
                values = list(radar_metrics.values())
                
                # 完成雷达图
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                ax.plot(angles, values, 'o-', linewidth=2, label='模型性能')
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_title('模型性能雷达图', size=14, y=1.05)
                ax.grid(True)
                ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "evaluation" / "plots" / "metrics_radar.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"⚠️ 无法绘制指标雷达图: {e}")