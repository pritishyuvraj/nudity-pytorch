# nudity-pytorch
Nudity detection ML model in PyTorch 

# Nudity detection ML model in PyTorch
Welcome to the repository of our PyTorch model designed to detect nudity in images. This model is built on the philosophy that a simpler model with more data is always better than a larger model with less data.
# Model Details
ResNet (Residual Network) is a deep neural network architecture introduced by Microsoft Research in 2015. It was primarily developed for image classification tasks and has since become a widely used and highly successful model in computer vision. The original paper titled "Deep Residual Learning for Image Recognition" is available at https://arxiv.org/abs/1512.03385. </br></br>
![Residual Network Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20200424011138/ResNet.PNG)
The key innovation of ResNet is the introduction of residual connections, which allow the network to train deeper than previous architectures without suffering from vanishing gradient problems. These residual connections skip one or more layers in the network and add the output of those skipped layers to the input of the subsequent layers, creating a form of identity mapping that eases the training process.</br></br>
This involves utilizing a pre-trained model and subsequently freezing the layers while adding an additional layer specifically designed for fine-tuning. In doing so, I have generated two distinct published models: one fine-tuned on a relatively smaller dataset (```./trained_pytorch_model/resnet_finetuned_smaller_dataset.pth```) and another trained on a more extensive dataset (WIP).
# Model Performance
Our model has demonstrated high accuracy in its predictions. Here are some key performance metrics:
```
Confusion Matrix:
 [[3186    9]
 [  40 2969]]
Test Accuracy: 0.9921
```
```
Number of Positive Samples: 3076
Number of Negative Samples: 3177
Percentage of True Positives: 98.50%
Percentage of True Negatives: 99.31%
```
## How to Use
### Install Dependencies
Install the necessary dependencies using the following command:
```
pip install -r requirements.txt
```
### Run the Model
Use the following command to run the model:
```
inference_model = torch.load('./trained_pytorch_model/inference_nudity_model.pth)
inference_model.eval()
model_prediction = inference_model(input)
```

## Contributing
We welcome contributions to improve this model. Feel free to submit issues, fork the repository and send pull requests!
## License
This project is licensed under the terms of the MIT license.
## Contact
If you have any questions, feel free to reach out or submit an issue.
#### Pritish Yuvraj
#### Email: pritish.yuvi@gmail.com
#### Linkedin -> https://www.linkedin.com/in/pritishyuvraj/
Thank you for visiting this repository, we hope you find it useful!
