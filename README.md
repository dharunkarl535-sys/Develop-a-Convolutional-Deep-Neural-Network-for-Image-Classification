# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model:

<img width="835" height="427" alt="568142822-eaed7ea2-019b-40a8-99b5-34f8b79b91a4" src="https://github.com/user-attachments/assets/e4912a56-9005-425c-9e4c-b04b033ec89b" />


## DESIGN STEPS
### STEP 1: 

Load the dataset from the tensorflow library.


### STEP 2: 

Preprocess the dataset.

### STEP 3: 

Create and train your model.

### STEP 4: 

Include the training loss, validation loss vs iteration plot.

### STEP 5: 

Test the model for your handwritten scanned images.


### STEP 6: 

Create and train your model.



## PROGRAM:


### Name: P.Bhavankumar

### Register Number: 212225240026

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)

        return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: Ragul M ')
print('Register Number: '212225240026')
summary(model, input_size=(1, 28, 28))

# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
  print('Name: Bhavankumar P')
  print('Register Number: 212225240026')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

      print('Name:Ragul M ')
      print('Register Number:212225240026 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Train the model
train_model(model, train_loader)


```

### OUTPUT

## Training Loss per Epoch:


## Confusion Matrix:

<img width="941" height="766" alt="592237526-6f9c695c-addc-4831-be3e-1419a3a514b6" src="https://github.com/user-attachments/assets/fa902694-489f-4a3d-9345-f26cf7d5515a" />


## Classification Report:

<img width="640" height="435" alt="592237641-93b21d58-cbac-4145-9761-96d2c4982822" src="https://github.com/user-attachments/assets/b6f0d771-f37e-45c2-963e-09af2bc121ac" />


### New Sample Data Prediction:

<img width="482" height="527" alt="592200927-1c374947-d3cd-4465-b8e8-11621c0835db" src="https://github.com/user-attachments/assets/b839896e-8605-434e-8e6d-9e078311b9ad" />

## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
