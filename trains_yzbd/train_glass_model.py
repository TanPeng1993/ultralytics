from ultralytics import YOLOv10

model = YOLOv10("ts_yolov10x.yaml")
# If you want to finetune the model with pretrained weights, you could load the
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='defect_detection_data_0522.yaml', epochs=400, batch=9, imgsz=2048, device=[0, 1, 2])
