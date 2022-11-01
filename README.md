# 二元坤类器
---


![kun](kun.jpg)
![kun_classify](kun_predict.jpg)
![zhiyin](zhiyin.jpg)
![zhiyin](zhiyin_predict.jpg)


## English
---

- What is this?
    - This is a binary classifier that classifies between a human(Kunkun) and chickens(arbitrary)
- How to use this classifier?
    1. Installed the packages within requirements.txt
        ```python
           pip -r install requirements.txt
        ```
    2. Go to inference.py
    3. Change the path of the **torch.load** at the 35th line to the *absolute path* of the *kun_weight.pt* file in this repository
    4. Change **img_path** to the *absolute path* of the image you want to classify (the classification only work for that specific human and arbitrary chickens, but of course you can use unrelated images for your own entertaining purposes)

    5. Change **fontpath** to the *absolute path* of the *kun.ttf* under the *font* directory of this repository
    6. Run the python program and enjoy!
- How did I made it?
    - I am using a neural network for this classifier, so we need data.
        - Dataset Preparation: We will need a lot of images to train the neural network(and ideally, quality images). Therefore, inspired by some blog posts, I have written a small scraper that will fetch the images I need from Google.
    - My choice of neural network architecture is a pre-trained ResNet-18.
        - Since it is pre-trained, my model can converges quickly and learn better by transfer learning than starts from scratch.
        - Might be an overkill in this task given the distinction between the features of the human and chicken.
    - Problems/Future Improvements
        - Try out some smaller neural network architecture or even custom architecture to test the performance of the classifier. 
        - Since the images are fetch from the internet, there are a lot of uncertainty in image quality. For example, there are a significant amount of noise in the dataset such that a picture of a lamb was fetch to the dataset for chicken.
            - Possible Solution, develop a multi-class classifier. For example, I can trained a 3-class classifier with weights from this binary classifiers, and the class labels will be **Kunkun**, **Chickens**, **Other**. In this way, I can use the 3-class classifier to sort the images for the binary classifier.
        - Object Detection model that can identify the face of each class label, such as the face of Kunkun and use the output coordinates for some interesting purposes.

## 中文
---
欢迎使用二元坤类器。因为小黑子们经常把鸡，哦不，只因和坤坤搞混并引以为傲，特作此分类器来治治
小黑子们的眼睛。下次如果不确定某张图片是只因还是坤坤的话，请务必使用此分类器。使用教程请看英文处。