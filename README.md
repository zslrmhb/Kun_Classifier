# 二元坤类器
---
## Web Version has Now Release!!!
[Kun Classifier](https://zslrmhb-kun-classifier-main-ln6tzp.streamlit.app/)

---
![kun](Assets/kun.jpg)
![kun_classify](Assets/kun_predict.jpg)
![zhiyin](Assets/zhiyin.jpg)
![zhiyin](Assets/zhiyin_predict.jpg)


## English
---

- What is this?
    - This is a binary classifier that classifies between a human(Kunkun) and chickens(arbitrary)
    ---
- How to use this classifier?
    1. Installed the packages within requirements.txt
        ```python
           pip -r install requirements.txt
        ```
    2. Go to inference.py
    3. Download the *kun_weight.pt* of this repository at this link: https://drive.google.com/file/d/1BYjrLHCwFRyfqR8oPDTLrEXT6S1TppPX/view?usp=sharing
    4. Change **img_path** to the *absolute path* of the image you want to classify (the classification only work for that specific human and arbitrary chickens, but of course you can use unrelated images for your own entertaining 
    5. Run the python program and enjoy!
---
- Datasets:
    - Kunkun: https://drive.google.com/drive/folders/17kMVt1Vhnm-W0dCF0o67Hwh3RLSvbcbc?usp=sharing
    - Chickens: https://drive.google.com/drive/folders/12axBSmUVnfIBY81798LY4im0N-GYx7Um?usp=sharing
    ---
- How did I made it?
    - I am using a neural network for this classifier, so we need data.
        - Dataset Preparation: We will need a lot of images to train the neural network(and ideally, quality images). Therefore, inspired by some blog posts, I have written a small scraper that will fetch the images I need from Google.
        - Scraper: https://github.com/zslrmhb/Kun_Scraper
    - My choice of neural network architecture is a pre-trained ResNet-18.
        - Since it is pre-trained, my model can converges quickly and learn better by transfer learning than starts from scratch.
        - Might be an overkill in this task given the distinction between the features of the human and chicken.
    - Problems/Future Improvements
        - Try out some smaller neural network architecture or even custom architecture to test the performance of the classifier. 
        - Since the images are fetch from the internet, there are a lot of uncertainty in image quality. For example, there are a significant amount of noise in the dataset such that a picture of a lamb was fetch to the dataset for chicken.
            - Possible Solution, develop a multi-class classifier. For example, I can trained a 3-class classifier with weights from this binary classifiers, and the class labels will be **Kunkun**, **Chickens**, **Other**. In this way, I can use the 3-class classifier to sort the images for the binary classifier.
        - Object Detection model that can identify the face of each class label, such as the face of Kunkun and use the output coordinates for some interesting purposes.
---
## 中文
---
欢迎使用二元坤类器。因为小黑子们经常把鸡，哦不，只因和坤坤搞混并引以为傲，特作此分类器来治治
小黑子们的眼睛。下次如果不确定某张图片是只因还是坤坤的话，请务必使用此分类器。使用教程请看英文处。