from sahi.predict import predict

images_folder_list = [
    # 工业相机
    # r"D:\Source_Code\defect-identification\picture-enhance\assets\source2",

    # 华盛
    r"C:\Users\t7981\Desktop\hs_sample\px_1_pk",
    r"C:\Users\t7981\Desktop\hs_sample\px_2_pk",
    r"C:\Users\t7981\Desktop\hs_sample\px_3_pk",

    # 手机测试数据
    # r"/mnt/data1/zhuhongji/data/2024_06_22/aug/test/images/"
]

for image_folder in images_folder_list:
    result = predict(
        model_type="yolov10",
        model_path="model/0727_960_big_yolov10b.pt",
        model_device="cpu",  # "cpu" or 'cuda:0'
        model_confidence_threshold=0.1,
        source=image_folder,
        slice_height=960,
        slice_width=960,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
