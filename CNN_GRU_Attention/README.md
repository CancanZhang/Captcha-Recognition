# Captcha Recognition (Variable Character Length) with CNN+GRU+Attention	 

Environment: Python 2.7.6 Tensorflow 1.4.1 Keras 2.1.5 Tesla K80*1

### CNN+GRU+Attention

**Note:**

- Generate captcha, with 4-6 characters, using package captcha
  ![2](https://github.com/CancanZhang/Captcha-Recognition/blob/master/CNN_GRU_Attention/img/captcha.png)
 
- Adding a class to denote '' character.

- It has 95.44\% categorical accuracy on validation dataset. However, the accuracy is only 75\%.
![hist](https://github.com/CancanZhang/Captcha-Recognition/blob/master/CNN_GRU_Attention/img/hist.png)

- Prediction:
![predict](https://github.com/CancanZhang/Captcha-Recognition/blob/master/CNN_GRU_Attention/img/predict.png)

