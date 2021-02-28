
import sys
from yolo_deneme import Getir_yolo
import streamlit as st
from PIL import Image, ImageOps
import time
import cv2


class window():
        def __init__(self):
                
                st.write("""
                         # X-RAY IMAGE DETECTION
                         """
                         )
                self.file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
                
               
                image = Image.open(self.file)
                image=image.resize((100,100))
                st.image(image)
                if(st.button("İşlem Yap")):
                    self.sec(self.file)  
                
        def sec(self,file_name):
             
                self.photo_path = file_name 
           
                self.islem()
      
               
               
        def islem(self):
                    import tensorflow as tf
                    import cv2
                    from keras.preprocessing import image
                    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
                    import numpy as np
                    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
                    from keras.models import Sequential
                    from keras.layers import Dense
                    from keras.applications.vgg16 import VGG16
                    import matplotlib.pyplot as plt
                    from glob import glob
                    start = time.time()
                  
                    
                    class_names =  ["Kovid","Normal","Zaturre"]
                    #try:
                        model = tf.keras.models.load_model("/Vgg19_web/yolovgg19")
                        self.img=Getir_yolo(self.photo_path)
                        dim = (224, 224)      
                        self.img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
                        resim_crop = cv2.resize(self.img, dim, interpolation = cv2.INTER_AREA)
              
                        image1 = self.img          
                        dim = (224, 224)
                        image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
                        img = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
                        #img=image1.resize((224,224))
                        img = image.img_to_array(img)
                        img = np.expand_dims(img, axis=0)
                        prediction = model.predict(img)
                        
                        prediction = np.argmax(prediction,axis=1)
                        print(prediction)
                        print(class_names[prediction[0]])
                    
                        end = time.time()
                        fark = end - start   
                        st.write("Tahmin Sonucu: "+str(class_names[prediction[0]]))
                        st.write("Tahmin Süresi(sn): "+str(round(fark,2)))
                    #except:
                        #st.write("Ciğer Tespit Edilemedi")
            
if __name__ == "__main__":   
    window = window()         
           
    
