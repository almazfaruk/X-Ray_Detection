# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:06:55 2020

@author: ASUS
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
def Getir_yolo(resim_yol):
 
        img = Image.open(resim_yol)
        
        dim = (600, 600)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)       
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
     
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)
        
        labels = ["ciger","ciger2","ciger3"]
        
        
        colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
        colors = [np.array(color.split(",")).astype("int") for color in colors]
        colors = np.array(colors)
        # colors = np.tile(colors,(18,1))
        
        
        model=cv2.dnn.readNetFromDarknet("./spot_yolov4.cfg","./spot_yolov4_final.weights")
        
        layers = model.getLayerNames()
        output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
        
        model.setInput(img_blob)
        
        detection_layers = model.forward(output_layer)
        
        ids_list=[]
        boxes_list=[]
        confidence_list=[]
        
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]
                
                if confidence > 0.20:
                    
                    label = labels[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                    
                    start_x = int(box_center_x - (box_width/2))
                    start_y = int(box_center_y - (box_height/2))
                    
                    ids_list.append(predicted_id)
                    confidence_list.append(float(confidence))
                    boxes_list.append([start_x, start_y,int(box_width),int(box_height)])
                    
            
        crop_image= []
        a=0
        b=0
        c=0
        d=0
        max_ids=cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4)
        
        for max_id in max_ids:
            max_class_id=max_id[0]
            box=boxes_list[max_class_id]
            
            start_x=box[0]
            start_y=box[1]
            box_width=box[2]
            box_height=box[3]
            
            predicted_id=ids_list[max_class_id]
            label=labels[predicted_id]
            confidence=confidence_list[max_class_id]
        
        
            end_x = start_x + box_width
            end_y = start_y + box_height
                    
            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
                    
                    
            label = "{}: {:.2f}%".format(label, confidence*100)
            print("predicted object {}".format(label))
             
              
            cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)
           
            a=start_x
            b=end_x
            c=start_y
            d=end_y
            cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            if a<0:
                a=0
            if b>img_width:
                b=img_width
            if c<0:
                c=0
            if d>img_height:
                d=img_height
            crop_image = img[c:d, a:b] 
            # cv2.imshow("Detection Window", crop_image)
            # cv2.waitKey(0)
           
        return crop_image  
        # crop_image = img[a:b, c:d] 
        # cv2.imshow("Detection Window", crop_image)
        # cv2.waitKey(0)
        
        # cv2.imshow("Detection Window", img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
  
            





























































