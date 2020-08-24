import cv2, time
import argparse
import os
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def make_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])

    ])
    
    return transform

def emotion(num):
    emotions = ['neutral','anger','surprise','smile','sad']
    return emotions[num]

def get_face_emotion(file_name,net):
    img = cv2.imread(file_name,cv2.IMREAD_COLOR)
    transform = make_transform()
    result_img = img.copy()
    h, w, _ = result_img.shape
    blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)

      # inference, find faces
    detections = net.forward()
    num = 0
    # postprocessing
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            #cv2.imwrite(f'../data/new/{num}.jpg',result_img[y1:y2,x1:x2])
            crop = result_img[y1:y2,x1:x2]

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            if crop is None:
                continue

            crop =  Image.fromarray(crop.astype('uint8'), 'RGB')

            crop = transform(crop).unsqueeze(0)
            with torch.no_grad():
                crop = crop.to(device)

                out = model(crop)


                pred = torch.argmax(out,dim=-1)
                pred = pred.type(torch.IntTensor).detach().cpu()

            label = emotion(pred)

          # draw rects
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), int(round(h/150)), cv2.LINE_AA)
            cv2.putText(result_img, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
    return result_img
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_weight', default='models/weight_best.pt', type=str, help='model weight path')
	parser.add_argument('--save_path', default='test_result', type=str, help='path for inferenced images')
	parser.add_argument('--test_path', default='../data/assignment/test/img', type=str, help='test images path')
	
	args = parser.parse_args()
	
	model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
	state = torch.load(args.load_weight)
	model.load_state_dict(state)
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	model_path = 'models/opencv_face_detector_uint8.pb'
	config_path = 'models/opencv_face_detector.pbtxt'
	net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
	conf_threshold = 0.7
	
	save_path = args.save_path
	test_path = args.test_path
	
	for img in os.listdir(test_path):
            print(f'{img}')
            name = img[0:-4]+'.jpg'
            result_img = get_face_emotion(f'{test_path}/{img}',net)
            cv2.imwrite(f'{save_path}/{name}',result_img)
