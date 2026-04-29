from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO('best.pt')  # 加载训练好的最佳模型
    #metrics = model.val()  # 在验证集上评估模型

    # 验证模型
    model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data='mydataset/data.yaml',  # 数据集配置文件路径
        split='test',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=1,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device='cpu',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=4,  # 数据加载的工作线程数（每个DDP进程）
        #conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        #iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='runs/val',  # 项目名称（可选）
        name='exp', # 实验名称
        plots=True  # 在训练/验证期间保存图像
    )