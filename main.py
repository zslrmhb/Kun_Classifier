import streamlit as st

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import  matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw

import pandas as pd

import time


# music = """
#         <audio controls autoplay>
#         <source src="{}" type="audio/mp3">
#         </audio>
#         """.format(os.getcwd() + '/background.mp3')
# # st.markdown(os.getcwd() + '/background.mp3')
# sound = st.empty()
# st.markdown(music, unsafe_allow_html=True)
# time.sleep(2)
# sound.empty()

# some global variables

KUN_DIR = os.getcwd() + '/Dataset/kun'
CHICKEN_DIR = os.getcwd() + '/Dataset/zhiyin'
WEIGHT_DIR = os.getcwd() + "\kun_weight.pt"

# Kun Classifier
device = "cuda" if torch.cuda.is_available() else "cpu"
class Kun_Classifier:
  """
  Binary Classification Class
  """
  def __init__(self):
    """
    Init.
    """
    pass

  def inference(self, img_path):
    """
    inference 

    :params: img_path: the image path of the image for inference
    :returns: the inference result in terms of strings
    """
    kun = "鉴定为坤"
    chicken = "鉴定为只因"

    model = models.resnet18(pretrained=True)
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, 1)
    model = model.to(device)

    img = Image.open(img_path).convert('RGB') 

    transformations = transforms.Compose([transforms.Resize((224,224)),
                      transforms.ToTensor(),
                      transforms.Normalize(
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225],
                      ),                      
  ])
    img_tensor = transformations(img).reshape(1,3,224,224).to(device)


    model.load_state_dict(torch.load(WEIGHT_DIR, map_location=torch.device(device)))

    model.eval()

    pred = model(img_tensor)
    confidence = torch.sigmoid(pred)
    if confidence < 0.5:
      return kun, confidence
    else:
      return chicken, confidence

kuner = Kun_Classifier()











# Welcome Page
st.title("KUN-er Classifier")
st.caption('Welcome! | 欢迎各位小黑子们前来体验二元坤类器!')
st.text(""".git

    """)


# Data Visualization
st.header('Data Visualization')


def show_and_classify(user_choice, classify=False):

    if user_choice == 0:
        result = KUN_DIR + '/' + random.choice(os.listdir(KUN_DIR))
    elif user_choice == 1:
        result = CHICKEN_DIR + '/' + random.choice(os.listdir(CHICKEN_DIR))
    else:
        result = random.choice([KUN_DIR + '/' + random.choice(os.listdir(KUN_DIR)), CHICKEN_DIR + '/' + random.choice(os.listdir(CHICKEN_DIR))])
    
    img = cv2.imread(result)
    fig = plt.figure()
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(converted_img)
    plt.axis('off')
    st.pyplot(fig, clear_figure=True)

    if classify: st.write(kuner.inference(result))

    # st.subheader('Kun!')




desired_image = st.slider('Kun(0) or Chicken(1)?', 0, 1)
if (st.button("Get")): show_and_classify(desired_image)


# Classification


st.header('Classification')
desired_classify = st.slider('Kun(0) or Chicken(1) or Random(2)?', 0, 2, 1)
if (st.button("Classify")): show_and_classify(desired_classify, True)


