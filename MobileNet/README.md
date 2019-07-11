# Captcha Recognition with MobileNet	 

Environment: Python 2.7.6 Tensorflow 1.4.1 Keras 2.1.5 Tesla K80*2

### MobileNet

- Light weight deep neural networks

- Factorize a standard convolution into a depthwise convolution and a 1X1 convolution 
![mobilenet](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/mobilenet.jpeg)

- Input Size: $D_F * D_F * M$

- Output Size: $D_F * D_F * N$

- Depthwise convolution filters filter through the whole input channels, each channel has only a single filter.  Input size: $D_F * D_F * M$ Output size: $D_F * D_F * M$

- Pointwise Convolution create new features using $N$ different *1 * 1 * M* filters. Then the output size is $D_F * D_F * N$.

- Computational cost in standard convolution filters:

   $$D_K * D_K * M * N * D_F * D_F$$

- Computational cost in depthwise separable convolutions: 

  $$D_K * D_K * M * D_F * D_F + M * N * D_F*D_F$$


**Note:**

- Generate captcha, with 4-6 characters, using package captcha
   ![1](https://github.com/CancanZhang/Captcha-Recognition/blob/master/CNN_GRU_Attention/img/captcha.png)

- Adding a class to denote '' character.

- Categorical accuracy: 99.22%, accuracy: 95.31%![hist](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/hist_mobilenet.png)

- Prediction![predict](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/predict_mobilenet.png)

- We also use another method to generate capthas, using different font, font size, font color, rotation, distance from each characters and background colors. Plus, we also add salt pepper noise or Gaussian noise on the images.

  ![mock1](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/mock1.png)
  ![mock2](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/mock2.png)
  ![mock3](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/mock3.png)
  ![mock4](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/mock4.png)
  
- Similarly, our model still cannot do well on this dataset (0% accuracy). Prediction result: ![predict_mock](https://github.com/CancanZhang/Captcha-Recognition/blob/master/MobileNet/img/predict_mock.png)
