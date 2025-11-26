# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import json
import os
from GarbageDetector import GarbageDetector
from config import test_config, MODEL_CONFIG

def main():
    st.set_page_config(
        page_title="智能垃圾分类检测系统",
        page_icon="🗑️",
        layout="wide"
    )
    
    # 标题和介绍
    st.title("🗑️ 智能垃圾分类检测系统")
    st.markdown("""
    基于YOLOv8的智能垃圾检测与分类系统，自动识别垃圾并给出分类建议。
    """)
    
    # ==================== 侧边栏配置 ====================
    st.sidebar.header("🔧 系统设置")
    
    # 模型选择
    selected_model_name = st.sidebar.selectbox(
        "选择检测模型",
        options=list(MODEL_CONFIG.keys()),
        index=1,
        help="选择用于垃圾检测的预训练模型"
    )
    
    # 获取选中模型的配置
    model_config = MODEL_CONFIG[selected_model_name]
    
    
    # 根据选择的模型设置默认置信度阈值
    confidence_threshold = st.sidebar.slider(
        "检测置信度阈值",
        min_value=0.1,
        max_value=0.9,
        value=model_config["default_confidence"],
        help=f"{selected_model_name}的推荐置信度阈值为{model_config["default_confidence"]}"
    )
    
    
    # ==================== 初始化检测器 ====================
    @st.cache_resource
    def load_detector(model_path):
        try:
            detector = GarbageDetector(model_path)
            return detector
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return None
    
    detector = load_detector(model_config["path"])
    
    if detector is None:
        st.warning("请确保模型路径正确，然后刷新页面")
        return
    
    # ==================== 主检测界面 ====================
    st.header("📸 垃圾检测")
    
    # 图像上传方式选择
    upload_method = st.radio(
        "选择图像输入方式:",
        ["上传图片", "使用示例图片", "摄像头拍摄"],
        horizontal=True
    )
    
    image_input = None
    
    if upload_method == "上传图片":
        uploaded_file = st.file_uploader(
            "选择一张包含垃圾的图片", 
            type=['jpg', 'jpeg', 'png'],
            help="支持 JPG, JPEG, PNG 格式"
        )
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
            st.success("✅ 图像上传成功！")
    
    elif upload_method == "使用示例图片":
        image_input = test_config['example_img_path']
        st.info("示例图片功能需要预先准备示例图像文件")
    
    else:  # 摄像头拍摄
        st.info("请使用摄像头拍摄包含垃圾的照片")
        camera_image = st.camera_input("拍摄垃圾照片")
        if camera_image is not None:
            image_input = Image.open(camera_image)
            st.success("✅ 照片拍摄成功！")
    
    # ==================== 检测结果显示 ====================
    if image_input is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 原始图像")
            st.image(image_input, width='stretch', caption="原始输入图像")
        
        with col2:
            st.subheader("🔍 检测结果")
            
            if st.button("开始检测", type="primary"):
                with st.spinner("正在检测垃圾..."):
                    # 执行检测
                    results = detector.detect(image_input, confidence_threshold)
                    
                    # 过滤低置信度结果
                    filtered_detections = [
                        d for d in results['detections'] 
                        if d['confidence'] >= confidence_threshold
                    ]
                    results['detections'] = filtered_detections
                    results['total_count'] = len(filtered_detections)
                    
                    # 显示检测结果图像
                    st.image(results['annotated_image'], width='stretch')
                    
                    # 显示统计信息
                    st.subheader("📊 检测统计")
                    
                    # 创建统计图表
                    stats_data = []
                    for category, info in results['category_stats'].items():
                        if info['count'] > 0:
                            stats_data.append({
                                '垃圾分类': category,
                                '数量': info['count'],
                                '颜色': detector.category_colors.get(category, (0,0,0))
                            })
                    
                    if stats_data:
                        # 饼图
                        fig = px.pie(
                            stats_data, 
                            values='数量', 
                            names='垃圾分类',
                            title='垃圾分类分布',
                            color='垃圾分类',
                            color_discrete_map={
                                '可回收物': 'green',
                                '有害垃圾': 'red', 
                                '厨余垃圾': 'orange',
                                '其他垃圾': 'gray'
                            }
                        )
                        st.plotly_chart(fig, width='stretch')
                        st.metric("总检测数量", results['total_count'])
                    else:
                        st.warning("未检测到符合条件的垃圾物品")
                    
                    # 详细检测结果表格
                    st.subheader("📋 检测详情")
                    
                    if results['detections']:
                        # 创建结果表格
                        df_data = []
                        for detection in results['detections']:
                            df_data.append({
                                '物品名称': f"{detection['icon']} {detection['class']}",
                                '垃圾类别': detection['category'],
                                '置信度': f"{detection['confidence']:.3f}",
                                '处理建议': detection['advice']
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, width='stretch')
                    else:
                        st.warning("未检测到符合条件的垃圾物品")
            
            else:
                st.info("点击'开始检测'按钮进行分析")
    
    # ==================== 模型信息展示 ====================
    st.header("🔬 模型信息")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📊 网络结构", "📈 训练指标", "ℹ️ 模型详情"])
    
    with tab1:
        st.subheader("📊 网络结构信息")
        
        # 加载并显示网络结构
        structure_file = model_config.get("structure_file")
        if structure_file and os.path.exists(structure_file):
            try:
                with open(structure_file, 'r', encoding='utf-8') as f:
                    network_structure = json.load(f)
                
                # 显示网络结构统计
                total_layers = len(network_structure)
                total_params = sum(layer.get('params', 0) for layer in network_structure)
                
                # 使用columns创建美观的统计卡片
                st.markdown("### 模型概览")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("总层数", total_layers, help="神经网络总层数")
                with col2:
                    st.metric("总参数量", f"{total_params:,}", help="模型参数总量")
                with col3:
                    st.metric("模型类型", "YOLOv8s", help="使用的YOLO版本")
                with col4:
                    st.metric("检测头数", "3", help="多尺度检测头数量")
                
                # 创建美观的网络结构表格
                st.markdown("### 详细网络结构")
                
                # 准备表格数据
                table_data = []
                for i, layer in enumerate(network_structure):
                    # 解析参数信息
                    params = layer.get('params', 0)
                    module_name = layer.get('module', '').split('.')[-1]
                    arguments = layer.get('arguments', [])
                    
                    # 格式化参数
                    if module_name == 'Conv':
                        param_desc = f"in={arguments[0]}, out={arguments[1]}, kernel={arguments[2]}"
                    elif module_name == 'C2f':
                        param_desc = f"in={arguments[0]}, out={arguments[1]}, n={arguments[2]}"
                    elif module_name == 'Detect':
                        param_desc = f"classes={arguments[0]}, channels={arguments[1]}"
                    else:
                        param_desc = str(arguments)
                    
                    table_data.append({
                        "层索引": i,
                        "模块类型": module_name,
                        "参数数量": f"{params:,}",
                        "输入来源": str(layer.get('from', '')),
                        "参数描述": param_desc
                    })
                
                # 创建DataFrame并显示
                df_structure = pd.DataFrame(table_data)
                
                # 使用st.dataframe并添加样式
                st.dataframe(
                    df_structure,
                    width='stretch',
                    height=400,
                    column_config={
                        "层索引": st.column_config.NumberColumn(width="small"),
                        "模块类型": st.column_config.TextColumn(width="medium"),
                        "参数数量": st.column_config.TextColumn(width="medium"),
                        "输入来源": st.column_config.TextColumn(width="small"),
                        "参数描述": st.column_config.TextColumn(width="large")
                    }
                )
                
                # 添加层类型统计
                st.markdown("### 层类型分布")
                layer_types = df_structure['模块类型'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 创建层类型饼图
                    fig_layers = px.pie(
                        values=layer_types.values,
                        names=layer_types.index,
                        title="网络层类型分布",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_layers.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_layers, width='stretch')
                
                with col2:
                    # 显示统计信息
                    st.markdown("#### 统计摘要")
                    for layer_type, count in layer_types.items():
                        st.write(f"**{layer_type}**: {count}层")
                    
                    st.metric("平均参数量/层", f"{total_params//total_layers:,}")
                        
            except Exception as e:
                st.error(f"加载网络结构失败: {e}")
        else:
            st.warning("网络结构文件不存在")
    with tab2:
        st.subheader("📈 训练评估指标")
        
        metrics_dir = model_config.get("metrics_dir")
        if metrics_dir and os.path.exists(metrics_dir):
            # 定义指标图片和描述
            metrics_config = {
                "results.png": {
                    "title": "训练结果曲线",
                    "description": "显示训练过程中的损失函数和mAP指标变化"
                },
                "confusion_matrix.png": {
                    "title": "混淆矩阵", 
                    "description": "显示模型在各类别上的分类混淆情况"
                },
                "confusion_matrix_normalized.png": {
                    "title": "归一化混淆矩阵",
                    "description": "按行归一化的混淆矩阵，显示召回率"
                },
                "BoxF1_curve.png": {
                    "title": "F1分数曲线",
                    "description": "不同置信度阈值下的F1分数变化"
                },
                "BoxP_curve.png": {
                    "title": "精确率曲线",
                    "description": "不同置信度阈值下的精确率变化"
                },
                "BoxR_curve.png": {
                    "title": "召回率曲线", 
                    "description": "不同置信度阈值下的召回率变化"
                },
                "BoxPR_curve.png": {
                    "title": "P-R曲线",
                    "description": "精确率-召回率曲线，曲线下面积表示AP"
                },
                "labels.jpg": {
                    "title": "标签分布",
                    "description": "训练数据集中各类别边界框的尺寸和位置分布"
                }
            }
            
            # 创建两列布局
            col_left, col_right = st.columns(2)
            
            with col_left:
                # 第一组指标
                st.markdown("#### 📊 训练过程指标")
                
                # 训练结果曲线
                results_path = os.path.join(metrics_dir, "results.png")
                if os.path.exists(results_path):
                    st.image(results_path, width='stretch', 
                            caption=metrics_config["results.png"]["description"])
                else:
                    st.info("训练结果曲线暂不可用")
                
                # 混淆矩阵
                st.markdown("#### 🎯 分类性能指标")
                cm_col1, cm_col2 = st.columns(2)
                
                with cm_col1:
                    cm_path = os.path.join(metrics_dir, "confusion_matrix.png")
                    if os.path.exists(cm_path):
                        st.image(cm_path, width='stretch',
                                caption=metrics_config["confusion_matrix.png"]["description"])
                
                with cm_col2:
                    cm_norm_path = os.path.join(metrics_dir, "confusion_matrix_normalized.png")
                    if os.path.exists(cm_norm_path):
                        st.image(cm_norm_path, width='stretch',
                                caption=metrics_config["confusion_matrix_normalized.png"]["description"])
            
            with col_right:
                # 第二组指标
                st.markdown("#### 📈 检测性能曲线")
                
                # 创建标签页来组织相关曲线
                curve_tab1, curve_tab2, curve_tab3 = st.tabs(["F1曲线", "P-R曲线", "其他曲线"])
                
                with curve_tab1:
                    f1_path = os.path.join(metrics_dir, "BoxF1_curve.png")
                    if os.path.exists(f1_path):
                        st.image(f1_path, width='stretch',
                                caption=metrics_config["BoxF1_curve.png"]["description"])
                    else:
                        st.info("F1曲线暂不可用")
                
                with curve_tab2:
                    pr_path = os.path.join(metrics_dir, "BoxPR_curve.png")
                    if os.path.exists(pr_path):
                        st.image(pr_path, width='stretch',
                                caption=metrics_config["BoxPR_curve.png"]["description"])
                    else:
                        st.info("P-R曲线暂不可用")
                
                with curve_tab3:
                    col_p, col_r = st.columns(2)
                    with col_p:
                        p_path = os.path.join(metrics_dir, "BoxP_curve.png")
                        if os.path.exists(p_path):
                            st.image(p_path, width='stretch',
                                    caption=metrics_config["BoxP_curve.png"]["description"])
                    
                    with col_r:
                        r_path = os.path.join(metrics_dir, "BoxR_curve.png")
                        if os.path.exists(r_path):
                            st.image(r_path, width='stretch',
                                    caption=metrics_config["BoxR_curve.png"]["description"])
                
                # 标签分布
                st.markdown("#### 📋 数据分布分析")
                labels_path = os.path.join(metrics_dir, "labels.jpg")
                if os.path.exists(labels_path):
                    st.image(labels_path, width='stretch',
                            caption=metrics_config["labels.jpg"]["description"])
                else:
                    st.info("标签分布图暂不可用")
            
            # 添加指标解读说明
            with st.expander("💡 指标解读指南", expanded=False):
                st.markdown("""
                **指标说明**:
                - **mAP**: 平均精度均值，综合评估检测性能，值越高越好
                - **混淆矩阵**: 显示模型分类正确和错误的情况
                - **P-R曲线**: 曲线下面积(AP)越大，检测性能越好  
                - **F1分数**: 精确率和召回率的调和平均数
                - **标签分布**: 显示训练数据的边界框分布特征
                """)
                
        else:
            st.info("该模型暂无训练指标数据")
            # 提供占位图示例
            st.markdown("#### 指标展示示例布局")
            example_col1, example_col2 = st.columns(2)
            
            with example_col1:
                st.info("训练曲线将显示在这里")
                st.info("混淆矩阵将显示在这里")
            
            with example_col2:
                st.info("性能曲线将显示在这里")
                st.info("数据分布将显示在这里")
    with tab3:
        st.subheader("ℹ️ 模型配置详情")
        
        # 使用卡片式布局展示模型信息
        st.markdown("### 模型基本信息")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.markdown("#### 🆔 身份信息")
            st.write(f"**模型名称**: {selected_model_name}")
            st.write(f"**模型路径**: `{model_config['path']}`")
            st.write(f"**模型描述**: {model_config['description']}")
        
        with info_col2:
            st.markdown("#### ⚙️ 配置信息")
            st.write(f"**默认置信度**: `{model_config['default_confidence']}`")
            st.write(f"**输入尺寸**: `640×640`")
            st.write(f"**类别数量**: `9`")
            st.write(f"**参数量**: `11,129,067`")
            st.write(f"**计算量**: `28.5 GFLOPs`")
        
        with info_col3:
            st.markdown("#### 📁 文件状态")
            structure_status = "✅ 已加载" if structure_file and os.path.exists(structure_file) else "❌ 缺失"
            metrics_status = "✅ 已加载" if metrics_dir and os.path.exists(metrics_dir) else "❌ 缺失"
            model_status = "✅ 已加载" if detector is not None else "❌ 缺失"
            
            st.write(f"**网络结构**: {structure_status}")
            st.write(f"**训练指标**: {metrics_status}")
            st.write(f"**模型权重**: {model_status}")
        
        # 性能指标展示
        st.markdown("### 🎯 性能指标")
        
        # 根据模型类型显示不同的性能数据
        if "全量微调" in selected_model_name:
            performance_data = {
                "mAP@0.5": 0.459,
                "精确率": 0.667, 
                "召回率": 0.418,
                "F1分数": 0.514,  # 计算得出: 2*(0.667*0.418)/(0.667+0.418)
                "推理速度": "197.3ms"
            }
            # 各类别性能表格
            st.markdown("#### 📊 各类别性能详情")
            class_performance = {
                "类别": ["all", "Bottle", "Book", "Mobile phone", "Banana", "Apple", "Orange", "Plastic bag", "Toilet paper", "Coffee cup"],
                "精确率(P)": [0.667, 0.509, 0.459, 0.769, 0.513, 0.669, 0.342, 0.916, 1.000, 0.828],
                "召回率(R)": [0.418, 0.408, 0.349, 0.714, 0.115, 0.250, 0.307, 0.889, 0.000, 0.730],
                "mAP@0.5": [0.459, 0.397, 0.318, 0.784, 0.178, 0.343, 0.303, 0.898, 0.060, 0.849],
                "mAP@0.5:0.95": [0.370, 0.316, 0.203, 0.712, 0.113, 0.292, 0.245, 0.712, 0.034, 0.704]
            }
        elif "分阶段微调" in selected_model_name:
            performance_data = {
                "mAP@0.5": 0.435,
                "精确率": 0.557,
                "召回率": 0.431,
                "F1分数": 0.487,  # 计算得出: 2*(0.557*0.431)/(0.557+0.431)
                "推理速度": "204.7ms"
            }
            
            # 各类别性能表格
            st.markdown("#### 📊 各类别性能详情")
            class_performance = {
                "类别": ["all", "Bottle", "Book", "Mobile phone", "Banana", "Apple", "Orange", "Plastic bag", "Toilet paper", "Coffee cup"],
                "精确率(P)": [0.557, 0.458, 0.437, 0.728, 0.277, 0.540, 0.270, 0.443, 1.000, 0.861],
                "召回率(R)": [0.431, 0.462, 0.362, 0.746, 0.154, 0.312, 0.273, 0.778, 0.000, 0.794],
                "mAP@0.5": [0.435, 0.425, 0.315, 0.737, 0.195, 0.321, 0.200, 0.766, 0.103, 0.852],
                "mAP@0.5:0.95": [0.340, 0.332, 0.194, 0.653, 0.113, 0.254, 0.154, 0.594, 0.103, 0.668]
            }
        else:
            performance_data = {
                "mAP@0.5": 0.250,
                "精确率": 0.320,
                "召回率": 0.280,
                "F1分数": 0.298,
                "推理速度": "50 FPS"
            }
        
        # 创建性能指标卡片
        st.markdown("#### 整体性能")
        perf_cols = st.columns(5)
        metrics = list(performance_data.items())
        
        for i, (metric_name, metric_value) in enumerate(metrics):
            with perf_cols[i]:
                if isinstance(metric_value, float):
                    display_value = f"{metric_value:.3f}"
                else:
                    display_value = metric_value
                
                st.metric(
                    label=metric_name,
                    value=display_value,
                    delta=None
                )
        
        # 显示各类别详细性能
        st.markdown("#### 各分类性能")
        if class_performance:
            df_class_perf = pd.DataFrame(class_performance)
            st.dataframe(
                df_class_perf,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "类别": st.column_config.TextColumn(width="medium"),
                    "精确率(P)": st.column_config.NumberColumn(format="%.3f"),
                    "召回率(R)": st.column_config.NumberColumn(format="%.3f"),
                    "mAP@0.5": st.column_config.NumberColumn(format="%.3f"),
                    "mAP@0.5:0.95": st.column_config.NumberColumn(format="%.3f")
                }
            )
        
        # 性能分析
        st.markdown("#### 📈 性能分析")
        if "全量微调" in selected_model_name:
            st.success("""
            **优势**: 
            - 在Plastic bag、Coffee cup和Mobile phone类别上表现优秀(mAP@0.5 > 0.75)
            - 整体精确率较高(0.667)
            - 推理速度相对较快
            """)
            st.warning("""
            **改进空间**:
            - Banana和Toilet paper类别的召回率较低
            - 整体召回率(0.418)有提升空间
            """)
        elif "分阶段微调" in selected_model_name:
            st.success("""
            **优势**: 
            - 在Coffee cup类别上表现最佳(mAP@0.5=0.852)
            - Mobile phone和Plastic bag类别表现良好
            - 整体召回率相对均衡
            """)
            st.warning("""
            **改进空间**:
            - Orange类别的精确率和召回率都较低
            - 整体精确率(0.557)需要提升
            - 推理速度稍慢
            """)
    # ==================== 垃圾分类指南 ====================
    st.header("📚 垃圾分类指南")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("♻️ 可回收物")
        st.markdown("""
        - 塑料瓶
        - 书本纸张  
        - 玻璃制品
        - 金属罐
        - 干净包装
        """)
    
    with col2:
        st.subheader("☣️ 有害垃圾") 
        st.markdown("""
        - 电池
        - 电子产品
        - 过期药品
        - 化学品
        - 荧光灯管
        """)
    
    with col3:
        st.subheader("🍎 厨余垃圾")
        st.markdown("""
        - 食物残渣
        - 果皮果核
        - 茶叶咖啡渣
        - 过期食品
        - 花草植物
        """)
    
    with col4:
        st.subheader("⚫ 其他垃圾")
        st.markdown("""
        - 污染的塑料
        - 卫生纸
        - 一次性餐具
        - 陶瓷碎片
        - 毛发灰尘
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "智能垃圾分类检测系统 | "
        "基于YOLOv8目标检测 | "
        "助力环保，从我做起 🌍"
    )

if __name__ == "__main__":
    main()