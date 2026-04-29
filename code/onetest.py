from ultralytics import YOLO
import cv2
import numpy as np

def draw_detect_results(frame, detections,classnames):
    """
    在图像上绘制跟踪结果
    Args:
        frame: 原始图像
        detections: 检测到的目标信息
        classnames: 每类的名称
    Returns:
        annotated_frame: 标注后的图像
    """
    annotated_frame = frame.copy()
    
    # 绘制每个检测到的目标
    for obj in detections:
        x1, y1, x2, y2 = obj[:4] #坐标信息
        conf = obj[4] #检测的置信度
        cls_id = obj[5]#类别
        
        #这一类对应的名称
        name = classnames[cls_id]
        
        
        # 画出矩形位置
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0))
        
        label = f"{name} {conf:.2f}" #要显示的标签文本
        #label = f"{name}"
        #矩形上方显示标签文本
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
       

    
    return annotated_frame

#这个程序 读取一张图片测试结果
if __name__ == '__main__':
    model = YOLO('best.pt')  # 加载训练好的最佳模型
   
    model.eval()
    
    #要测试的图像文件名称
    #picfilename = 'mydataset\\test\\images\\DSC_1243_17kv1r22k_0.jpg'
    picfilename = 'mydataset\\test\\images\\DSC_1046_17kv10r3k_8.jpg'
    image = cv2.imdecode(np.fromfile(picfilename, dtype=np.uint8), -1)


    if image is None:
       print('图像读取失败，请检查图像名称路径是否有误！')

    if len(image.shape) == 2:
        image = cv2.merge([image, image, image])
    
    results = model(image,conf=0.25,verbose=False)[0]
  
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
           # 获取检测框信息
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
           
            #存入到检测列表内
            detections.append([x1, y1, x2, y2, conf, cls_id])
   
    #检测的结果绘制到原图上
    annotated_frame = draw_detect_results(image,detections,model.names)
    cv2.imshow("Detection", annotated_frame)
    cv2.waitKey(0)