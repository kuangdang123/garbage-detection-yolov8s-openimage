train_config = {
    "model_path" : "yolov8s-oiv7.pt",
    "trian_dataset_path" : "./export_oi/dataset.yaml"
}

test_config = {
    "model_path" : "garbage_detection_precise/stage4_full_finetune/weights/best.pt",
    "img_path" : "export_oi/images/test/0ec4b1c27046c4ca.jpg",
    "example_img_path": "export_oi/images/test/00b729b5187a1898.jpg",
    "yaml_path" : "export_oi/dataset.yaml",
    "batch_imgs_path" : "export_oi/images/test",
    "test_output_dir" : "test/",
}

MODEL_CONFIG = {
    "YOLOv8s 全量微调": {
        "path": "garbage_detection/yolov8s_finetuned/weights/best.pt",
        "default_confidence": 0.634,
        "description": "使用完整数据集进行端到端微调的模型",
        "structure_file": "network_structure.json",
        "metrics_dir": "garbage_detection/yolov8s_finetuned/"
    },
    "YOLOv8s 分阶段微调": {
        "path": "garbage_detection_precise/stage4_full_finetune/weights/best.pt",
        "default_confidence": 0.430,
        "description": "采用分层解冻策略进行精细微调的模型",
        "structure_file": "network_structure.json",
        "metrics_dir": "garbage_detection_precise/stage4_full_finetune/"
    },
    "YOLOv8s 预训练模型": {
        "path": "yolov8s-oiv7.pt",
        "default_confidence": 0.25,
        "description": "原始COCO预训练模型，未进行垃圾检测微调",
        "structure_file": "network_structure.json",
        "metrics_dir": None
    }
}

GARBAGE_CLASSIFICATION = {
    # 可回收物
    "Bottle": {
        "name" : "瓶子",
        "category": "可回收物",
        "color": (0, 255, 0),  # 绿色
        "advice": "请清空内容物，可以压扁后投入可回收物垃圾桶",
        "icon": "♻️"
    },
    "Book": {
        "name": "书本",
        "category": "可回收物", 
        "color": (0, 255, 0),
        "advice": "保持干燥整洁，投入可回收物垃圾桶",
        "icon": "♻️"
    },
    
    # 有害垃圾
    "Mobile phone": {
        "name": "手机",
        "category": "有害垃圾",
        "color": (255, 0, 0),  # 红色
        "advice": "含有重金属，请投入有害垃圾回收箱或专门回收点",
        "icon": "☣️"
    },
    
    # 厨余垃圾  
    "Banana": {
        "name": "香蕉",
        "category": "厨余垃圾",
        "color": (255, 165, 0),  # 橙色
        "advice": "请投入厨余垃圾桶，可用于堆肥",
        "icon": "🍌"
    },
    "Apple": {
        "name": "苹果",
        "category": "厨余垃圾",
        "color": (255, 165, 0),
        "advice": "果核可降解，请投入厨余垃圾桶",
        "icon": "🍎"
    },
    "Orange": {
        "name": "橙子",
        "category": "厨余垃圾", 
        "color": (255, 165, 0),
        "advice": "果皮易腐烂，请投入厨余垃圾桶",
        "icon": "🍊"
    },
    
    # 其他垃圾
    "Plastic bag": {
        "name": "塑料袋",
        "category": "其他垃圾",
        "color": (128, 128, 128),  # 灰色
        "advice": "污染的塑料袋属于其他垃圾，请投入其他垃圾桶",
        "icon": "🛍️"
    },
    "Toilet paper": {
        "name": "厕纸",
        "category": "其他垃圾",
        "color": (128, 128, 128),
        "advice": "使用过的卫生纸属于其他垃圾，请投入其他垃圾桶", 
        "icon": "🧻"
    },
    "Coffee cup": {
        "name": "咖啡杯",
        "category": "其他垃圾",
        "color": (128, 128, 128),
        "advice": "一次性咖啡杯通常属于其他垃圾，请投入其他垃圾桶",
        "icon": "☕"
    }
}