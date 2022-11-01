import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw

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

    #TODO
    # model.load_state_dict(torch.load("???.pt", map_location=torch.device(device)))

    # model.load_state_dict(torch.load("C:/Users/zslrm/Desktop/Data_Science_Projects/Kun_Classifier/kun_weight.pt", map_location=torch.device(device)))
    model.eval()

    pred = model(img_tensor)
    if torch.sigmoid(pred) < 0.5:
      return kun
    else:
      return chicken

  def classify_and_show(self, img_path, fontpath):
    """
    Output Classification Result and show it in a window

    :params: img_path 
    :params: fontpath 

    """
    inference_result = self.inference(img_path)
    

    dimensions = (224, 224)
    height = dimensions[0]
    width = dimensions[1]
    center = (width // 2 - 110, height // 2 + 100)

    picture = cv2.imread(img_path)
    picture = cv2.resize(picture, dimensions)

    font_scale = 0.5
    font = ImageFont.truetype(fontpath, 45)
    fill = (0, 0, 255, 0)

    # we will use pillow since OpenCV does not support Chinese Characters
    img_pil = Image.fromarray(picture)
    draw = ImageDraw.Draw(img_pil)
    draw.text((center), inference_result, font=font, fill=fill)
    img = np.array(img_pil)

    cv2.imshow(inference_result ,img)

    # cv2.imwrite('zhiyin_predict.jpg', img)
    cv2.waitKey(0)


#TODO
#img_path = ???.jpg
#fontpath = ???.ttf

# img_path = 'C:/Users/zslrm/Desktop/Data_Science_Projects/Kun_Classfier/zhiyin.jpg'
# fontpath = "C:/Users/zslrm/Desktop/Data_Science_Projects/Kun_Classfier/font/kun.ttf"


kuner = Kun_Classifier()
kuner.classify_and_show(img_path, fontpath)