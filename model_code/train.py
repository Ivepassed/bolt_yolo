from ultralytics import YOLO
import os
if __name__ == '__main__':
    # Load a model
    # model = YOLO("E:\\ultralytics-main\\yolov5n.pt")
    model = YOLO("E:\\ultralytics-main\\yolo11n.pt")
    # model = YOLO("E:\\ultralytics-main\\runs\\detect\\chicken\\train3\\weights\\best.pt")

    train_results = model.train(
        # data="E:\\ultralytics-main\\datasets\\xxx\\data.yaml",  # path to dataset YAML
        data="E:\\ultralytics-main\\finish_dataset\\blind_sidewalk(DST1011)\\data.yaml",  # path to dataset YAML\\data.yaml",  # path to dataset YAML
        epochs=50,  # number of training epochs
        imgsz=640,  # training image size
        device='cpu',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,
        amp=False,
    )

    # # Evaluate model performance on the validation set
    # metrics = model.val()

    # Perform object detection on an image
    # results = model("E:\\ultralytics-main\\datasets\\aerail_vehicle\\images\\test\\LIS_600_240410_SO101_6-1KM_10H_mp4-0055_jpg.rf.4ec6aeb02103525734a8be8585584f3f.jpg")
    # results[0].show()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model