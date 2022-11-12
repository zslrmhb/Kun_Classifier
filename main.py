import streamlit as st

import os
import cv2
import random
import base64
import torch
import torch.nn as nn
import  matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw


KUN_DIR = os.getcwd() + '/Dataset/kun'
CHICKEN_DIR = os.getcwd() + '/Dataset/zhiyin'
WEIGHT_DIR = os.getcwd() + "/kun_weight.pt" 

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
    kun = "KUN | 含坤量为:{}%"
    chicken = "CHICKEN | 含只因量为:{}%"

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
    sigmoid = torch.sigmoid(pred)

    if sigmoid < 0.5:
      return kun.format(round(float(100 * (1-sigmoid)), 2))
    else:
      return chicken.format(round(float(100 * sigmoid), 2))

kuner = Kun_Classifier()

# Welcome Page
st.title("KUN-er Classifier")
st.caption('Welcome! | 欢迎各位小黑子们前来体验二元坤类器! | https://github.com/zslrmhb/Kun_Classifier')
st.text("""

    """)

def autoplay_audio(file_path: str):
  # code from https://github.com/streamlit/streamlit/issues/2446
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" loop="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.empty()
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
if st.button("Don't Click! | 小黑子勿按！"):autoplay_audio("background.mp3")


tab1, tab2, tab3 = st.tabs(["Data Visualization | 让我康康", "Classification | 二元坤类器", "Let me try try | 让我试试"])

# Data Visualization
with tab1:
  st.subheader('Data Visualization | 让我康康')

  def show_and_classify(user_choice, classify=False):

      if user_choice == 0:
          result = KUN_DIR + '/' + random.choice(os.listdir(KUN_DIR))
      elif user_choice == 1:
          result = CHICKEN_DIR + '/' + random.choice(os.listdir(CHICKEN_DIR))
      else:
          result = random.choice([KUN_DIR + '/' + random.choice(os.listdir(KUN_DIR)), CHICKEN_DIR + '/' + random.choice(os.listdir(CHICKEN_DIR))])

      img = cv2.imread(result)
      converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      st.image(converted_img)

      if classify: st.caption(kuner.inference(result))

  desired_image = st.slider('Kun(0) or Chicken(1)? | 坤 或 只因', 0, 1)
  if (st.button("Get")): show_and_classify(desired_image)

with tab2:
# Classification
  st.subheader('Classification | 二元坤类器')
  desired_classify = st.slider('Kun(0) or Chicken(1) or Random(2)? | 坤 或 只因 或 两个都要 (bushi', 0, 2, 1)
  if (st.button("Classify")): show_and_classify(desired_classify, True)

with tab3:
# User Input
  st.subheader('Let me try try | 让我试试')
  def classify_user_input(picture):
    if picture:
      st.caption(kuner.inference(picture))

  picture = st.camera_input("")
  classify_user_input(picture)