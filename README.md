# nudity-pytorch
Nudity detection ML model in PyTorch 

# Nudity detection ML model in PyTorch
Welcome to the repository of our PyTorch model designed to detect nudity in images. This model is built on the philosophy that a simpler model with more data is always better than a larger model with less data.
# Model Performance
Our model has demonstrated high accuracy in its predictions. Here are some key performance metrics:
```
Confusion Matrix:
 [[3155   46]
 [  22 3030]]
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
