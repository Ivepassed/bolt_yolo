from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("E:\\ultralytics-main\\yolov5n.pt")
    # model = YOLO("E:\\ultralytics-main\\yolo11n.pt")
    model = YOLO(".\\train\\weights\\best.pt")   #模型位置

    results = model("1.jpg") # #图片位置
    results[0].show()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model