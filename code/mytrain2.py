from ultralytics import YOLO
import yaml


if __name__ == '__main__':
    # 方式一：从配置文件构建模型（从头开始训练）
    model = YOLO('yolov8_SE.yaml')
    #model = YOLO('yolov8_SE.yaml').load('yolov8n.pt')

    # 方式二：加载预训练模型进行微调（推荐）
    #model = YOLO('yolov8n.pt')# 加载官方预训练权重

    # 开始训练
    results = model.train(
        data='mydataset/data.yaml',  # 数据集配置文件路径
        epochs=50,                # 训练轮数
        imgsz=640,                 # 输入图像尺寸
        batch=16,                  # 批次大小（根据GPU显存调整）
        device='cpu',                  # 使用GPU，如使用CPU则设为 None 或 'cpu'
        close_mosaic=10,           #关闭mosaic的次数
        workers=4,                 # 数据加载线程数
        project='runs/train',      # 结果保存目录 训练后的网络会存在这个目录内
        name='exp'                 # 实验名称 
    ) 
    
    

   