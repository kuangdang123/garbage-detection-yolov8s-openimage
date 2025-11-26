train_config = {
    "model_path" : "yolov8s-oiv7.pt",
    "trian_dataset_path" : "./export_oi/dataset.yaml"
}

test_config = {
    "model_path" : "garbage_detection_precise/stage4_full_finetune/weights/best.pt",
    "imgs_path" : "export_oi/images/test",
    "example_img_path": "export_oi/images/test/00b729b5187a1898.jpg",

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