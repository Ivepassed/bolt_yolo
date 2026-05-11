from ultralytics import YOLO
import torch

torch.cuda.init()
import os

if __name__ == '__main__':
    # Load a model
    # model = YOLO("model_code/yolov5n.pt")
    model = YOLO("model_code/yolo11n.pt")
    # model = YOLO("model_code/train/weights/best.pt")

    train_results = model.train(
        data="C:\\Users\\hillr\\Desktop\\bolt_yolo\\bolt.v7i.voc(DST2752)\\data.yaml",
        # path to dataset YAML\\data.yaml",  # path to dataset YAML
        epochs=50,  # number of training epochs
        imgsz=640,  # training image size
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        device=torch.device('cuda'),  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device='cpu',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
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
