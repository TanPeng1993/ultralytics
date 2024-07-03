from ultralytics import YOLO

# model = YOLOv10('/mnt/data1/tanpeng/model_train/yolov10/runs/detect/train10/weights/best.pt')
model = YOLO('model/best_yolov8.pt')
pic_list = [
    # r"/home/tanpeng/model_train/test_pic/industrial_camera_1/",
    # r"/home/tanpeng/model_train/test_pic/lab/",
    # r"/home/tanpeng/model_train/test_pic/0620/pk",
    # r"/home/tanpeng/model_train/test_pic/0620/ps"
    # Windows
    # r"D:\Source_Code\defect-identification\picture-enhance\assets\target2"
    # 华盛
    # r"C:\Users\t7981\Desktop\hs_sample\px_1_pk",
    # r"C:\Users\t7981\Desktop\hs_sample\px_1_ps",
    # r"C:\Users\t7981\Desktop\hs_sample\px_2_pk",
    r"C:\Users\t7981\Desktop\hs_sample\px_2_ps",
    # r"C:\Users\t7981\Desktop\hs_sample\px_3_pk",
]
for pic in pic_list:
    model.predict(source=pic, device=["cpu"], save=True, conf=0.01, iou=0.95)
