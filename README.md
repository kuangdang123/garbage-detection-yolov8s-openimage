# 垃圾分类检测系统
利用在coco预训练的模型yolov8s进行垃圾实时检测

## 文件结构
### config.py
定义基础配置、文件路径脚本

### yolov8s-oiv7.pt
经过COCO数据集预训练的模型本身，通过Fifty-One数据集管理工具进行下载访问[yolov8s](https://docs.voxel51.com/model_zoo/models/yolov8s_oiv7_torch.html)

### network_structure.json
模型的基本结构，包括网络层级结构及其参数和来源

### build_dataset.ipynb
构建数据集脚本，基于Fifty-One进行数据管理和导出，主要来源是大型开源数据集[Open Images Dataset V6](https://storage.googleapis.com/openimages/web/factsfigures.html)，筛选了四个大类共9个类别的垃圾数据：

```python
recyclables = ["Bottle", "Book"]
hazardous = ["Mobile phone"]
kitchen = ["Banana", "Apple", "Orange"]
residual = ["Plastic bag", "Toilet paper", "Coffee cup"]
```

目前数据集，训练集、验证集、测试集分别只有1000个样本，并以yolov8的训练格式进行导出（与yolov5训练格式兼容）

### train_yolov8s.ipynb, test_yolov8s.ipynb
进行训练、测试的脚本

### GarbageDetector.py
进行垃圾检测的工具类

### streamlit_app.py
app界面定义脚本，负责了前后端的交互


## 未来可能支持的部分

### 丰富数据集
- 增加数据种类，以及每类的数量
- 寻找更合理的数据，比如厨余垃圾，一般情况下显然不是一个完整的苹果或者香蕉作为垃圾，需要补充只有核/皮的情况，以及更为复杂的缺失部分、氧化变黑等情况，以适应生活情景
- 寻找更为本地化的数据，如特色外卖盒、饮料瓶等

### 添加一键配置 + 启动相关脚本，助力本地部署
#### Windows用户
1. 双击 `run_app.bat`
2. 等待依赖安装完成
3. 浏览器自动打开应用界面

#### Linux/Mac用户
1. 在终端运行: `chmod +x run_app.sh && ./run_app.sh`
2. 等待依赖安装完成  
3. 浏览器自动打开应用界面

#### Docker用户
```bash
docker-compose up
```
### 研究对比更多模型
比如`MobileNetV2`和`Faster R-CNN`
