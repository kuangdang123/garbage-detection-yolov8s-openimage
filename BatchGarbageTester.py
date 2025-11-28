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
        
        return self.results
    
    def _process_single_image_results(self, results, image_path, inference_time):
        """处理单张图片的检测结果"""
        image_results = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "inference_time": inference_time,
            "detections": [],
            "detection_count": 0,
            "categories_found": set()
        }
        
        for result in results:
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
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    
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