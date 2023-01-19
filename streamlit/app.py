#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import streamlit as st 

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import CustomObjectScope

def iou(y_true, y_pred):
        smooth = 1
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1
def dice_coef(y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

def f1(y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val



with CustomObjectScope({'f1':f1}):
  model_classify=tf.keras.models.load_model("chexnet_model.h5")
with CustomObjectScope({ 'iou': iou,'dice_coef': dice_coef, 'dice_loss': dice_loss}):
 model_segment=tf.keras.models.load_model("pneumo-model_aug.h5")
   
    
st.header("Diagnosis of Pneumothorax in ChestX-ray Images Using DeeplabV3+")

#examples=["107.png","108.png","109.png"]
#link='Check Out Our Github Repo ! [link](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)'
#st.markdown(link,unsafe_allow_html=True)





def load_image(image_file):
   image = Image.open(image_file)
   image = np.float32(image)
   size = 256
   image = cv2.resize(image,(256, 256))
   image = image / 255.0
   #st.write(image.shape)
   image= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
   #st.write(image.shape) 
   image = np.expand_dims(image, axis=0)
   #st.write(image.shape)
   return image
	 
	 
  
    
st.subheader("Upload a chest X-ray Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])




  
if image_file is not None:

      image=load_image(image_file)
      
      st.text("Making A Prediction ....")
      st.image(image,width=512)
      
      pred = model_classify.predict(image)[0]
      pred = np.argmax(pred)
      # if the probabiliy score is greater than 0.5 then give class label=1 else 0
      if pred >0.5:
        whether_pneumothorax = 1
      else:
        whether_pneumothorax = 0

      # if pneumothorax is predicted from classification model, then predict segmentation
      if whether_pneumothorax:
          st.subheader("THIS IMAGE CONTAINS PNEUMOTHORAX")
          image = np.squeeze(image, axis=0)
          image = cv2.resize(image, (512, 512))
          image = np.expand_dims(image, axis=0)
          # if the image contains pneumothorax, predict the mask segmentation
          pred_ms =  model_segment.predict(image)
          pred_mask = (pred_ms[0]>0.5).astype(np.uint8)
          #pred_mask= cv2.applyColorMap(pred_mask, cv2.COLORMAP_BONE)
          #pred_mask = cv2.imread(pred_mask,cv2.IMREAD_GRAYSCALE)
          st.write(pred_mask.shape)
          #pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
          fig = plt.figure(figsize = (15,15))
          plt.imshow(np.squeeze(image[0]),cmap='gray',alpha=1.0)
          plt.imshow(pred_mask,cmap='Reds',alpha=0.4)
          plt.axis("off")
          st.pyplot(fig)
          #st.image(pred_mask*255,width=850)

          st.text("DONE ! ....")
         

      else:
          st.subheader("THIS IMAGE DOES NOT CONTAIN PNEUMOTHORAX")
          st.image(image,width=512)
          st.text("DONE ! ....")

