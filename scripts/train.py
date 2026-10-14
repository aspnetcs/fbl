#coding:utf-8
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# Use the model # datasets/dataset/images/
if __name__ == '__main__':
    # Use the model
    results = model.train(data='E:/00code/3.06-code/class_Detection_end/datasets/dataset/data.yaml', epochs=250, batch=4)

    # success = model.export(format='onnx')



