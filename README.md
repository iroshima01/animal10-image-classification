# animal10-image-classification
## Convolutional Neural Networks (CNN) & Transfer Learning for Animal-10 Dataset

This project involves implementing a **Convolutional Neural Network (CNN) from scratch** and applying **Transfer Learning** using ResNet-18 and MobileNetV2 on the **Animal-10 Dataset**.

---

## **Part 1: Implementation of a Simple CNN Model**

### **1. Data Loading and Preprocessing**
- Load the **Animal-10 Dataset** using PyTorch’s `torchvision.datasets.ImageFolder`.
- Preprocess images by:
  - Resizing to **256x256**
  - Normalizing pixel values
  - Applying transformations using `torchvision.transforms`.
- Split data into:
  - **3000 training images** (300/class)
  - **750 validation images** (75/class)
  - **750 test images** (75/class)

### **2. CNN Model Architecture**
- Implement a **CNN with 5 convolutional layers**.
- Apply **ReLU activation** after each convolution.
- Use **max pooling** to reduce spatial dimensions.
- Flatten the output and pass it through **fully connected layers**.
- Use **softmax** for multi-class classification.

### **3. Training Process**
- Define an appropriate **loss function (Cross-Entropy Loss)**.
- Choose an **optimizer (SGD / Adam)** with tuned hyperparameters.
- Train the model using **mini-batch gradient descent**:
  - Implement forward pass to compute predictions.
  - Compute loss and backpropagate gradients.
  - Update model parameters using the optimizer.
- Monitor performance by evaluating on the **validation set** periodically.
- Train for at least **30 epochs** and experiment with **learning rates** and **batch sizes**.
- Select the best model based on **validation accuracy**.

### **4. Model Evaluation**
- Test the best model on the test set.
- Compute metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Visualize the **confusion matrix**.
- Analyze the model’s performance and discuss key findings.

---

## **Part 2: Transfer Learning and Comparative Analysis**

### **1. ResNet-18 Models**
- Load the **pre-trained ResNet-18 model** from `torchvision.models`.
- Modify the **final fully connected (FC) layer** to match the number of classes in Animal-10.
- Train **three versions** of ResNet-18:
  - **Base Model:** Train only the final FC layer while keeping ResNet-18 weights frozen.
  - **Second Model:** Unfreeze **convolutional blocks 3 & 4** and train them along with the FC layer.
  - **Third Model:** Unfreeze **all layers** and train the entire ResNet-18 model.

### **2. MobileNetV2 Models**
- Load the **pre-trained MobileNetV2 model** from `torchvision.models`.
- Modify the **final FC layer** to match the number of classes in Animal-10.
- Train **two versions** of MobileNetV2:
  - **Base Model:** Train only the final FC layer while keeping MobileNet weights frozen.
  - **Second Model:** Unfreeze **all layers** and train the entire MobileNetV2 model.

### **3. Comparison and Analysis**
- Evaluate each trained model using:
  - **Validation accuracy**
  - **Test accuracy**
  - **Confusion matrix**
- Compare CNN vs. Transfer Learning models.
- Analyze results and discuss challenges faced during training.

---
