# Dataset
[The EEG-ImageNet dataset](https://github.com/perceivelab/eeg_visual_classification?tab=readme-ov-file)
- The EEG-ImageNet dataset has been recorded using a 128-channel cap (actiCAP 128Ch). The EEG-ImageNet includes the EEG signals of six human subjects produced as the result of visual stimulation.
- Contains 11,965 EEG sequences recorded while a group of 6 participant subjects looked at images displayed on a computer screen.
- The images are taken from a subset of 40 classes from ImageNet with 50 images per class (in total, 2,000 images).
- Each EEG sample has 128 channels, each one with about 500 values for each observed image.
- All EEG signals were first detrended in order to remove unwanted linear trends, filtered with a notch filter (50 Hz and its harmonics) and with a bandpass filter with low and high-cut frequencies, respectively, at 5 Hz and 95 Hz, and finally z-scored (zero-centered values with unitary standard
- Training, validation and test splits of the EEG dataset consists respectively of 1600 (80%), 200 (10%), 200 (10%) images with associated EEG signals, ensuring that all signals related to a given image belong to the same split.
- Eye fixations recorded using a 60-Hz Tobii T60 eye-tracker – of the same six human subjects who underwent EEG recording. Training, validation and test splits are the same of EEG data.

# Steps
![steps](https://github.com/user-attachments/assets/5aba1387-f8f6-43e5-943f-eed0f37e522f)


# Chebyshev Graph Convolution
Based on functional connectivity implemented with [torch.conv.ChebConv](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.ChebConv.html)

# Generative Deep Network
Generative deep modeling is considered as an unsupervised learning task that discovers and learns the contents in input data in such a way that the extracted model can be used to generate new examples that could have been extracted plausibly from the original dataset.

Extracts discriminative features of the different categories that the input belongs to.
  
- ![image](https://github.com/user-attachments/assets/7bff4f79-4cda-451a-9a00-e001bb1d285b)
- Includes four layers of graph convolution.
- The Laplacian of the input graph is necessary to estimate the graph convolution of the input in each layer.
- The estimation is performed via the Chebyshev polynomial expansion of the Laplacian graph.
- Then, a batch normalization filters the output of each layer.
- After the fourth graph convolution layer, the extracted feature vector is passed through a dropout layer.
- Then, the flattened output of the dropout layer is fed to a dense fully connected layer, and a log-softmax function is used for the classification of the output of the fully connected layer.
- Shows The weights are trained to classify 40 categories of image stimulation and the flattened vector before the last dense layer is used to impose to the next GAN part of the network. The dimension of the flattened vector is equal to 6400.
- ![image](https://github.com/user-attachments/assets/4c010da5-314f-4229-9e1a-a14330eebcc2)

- Each of the recorded EEG signals includes 128 channels and every node in the constructed graph includes 440 samples.
- The input dimension of the graph convolutional layer independent of the number of graph nodes is considered to be 440, equal to the number of samples in each node.
- The obtained graph with the first graph convolution has 128 nodes with 440 samples in each vertex.
- A graph with 128 nodes with 220 samples in each vertex is the output of the second graph convolution, and the output of the third graph convolution operation is a 128-node graph with 110 samples in each of the nodes, and accordingly, the graph output of the fourth layer has 50 samples in each node.
- The attained 128-node graph with 50 samples in each node outputs a vector with 6400 elements.
- The flattened vector is passed through a dense layer and the dimensions of the inputs and outputs of the dense layer are 6400 and 40, respectively.
- ![image](https://github.com/user-attachments/assets/a482185e-b43d-4de6-a5be-9b137bfb9349)
- Table 1 shows the dimensions of weight tensors for different layers of GDN part of the proposed GDN-GAN. Moreover, it shows the total number of parameters of graph convolutional layers according to the order of the Chebyshev polynomial expansion considered for each layer.

**Training Procedure**: In order to fit the proposed GDN part to the EEG-ImageNet dataset, a training procedure is implemented, and the parameter weights of the network are optimized.
  - A 10-fold cross-validation strategy is used to train and evaluate the proposed network.
  - A standard gradient descent (SGD) is used to optimize the proposed GDN in each iteration, and the optimum parameters of the GDN are determined with the convergence of the train and test accuracy.
  - The trained weights of the GDN are transfered to the GDN-GAN to train the reconstruction part of the network.

--------------------------
 # Generative Adversarial Network

The trained weight vectors of the network parameters are used as initial weight vectors to train the network to map the EEG signal to the image stimuli and realize the image reconstruction from the brain activity.

The GAN part maps the extracted feature vector to the image saliency.
  - ![image](https://github.com/user-attachments/assets/32bf9274-27fd-4b9a-97d6-9af465323709)

  - Figure 5 illustrates different layers of the GAN part of the proposed network.
## The Generator Part
  - The generator part of the GAN consists of two dense layers, followed by four sequential transposed two-dimensional (2D) convolution layers, and one 2D convolution layer and leaky rectified linear unit is used as the activation function of all layers except for the first dense layer.
  - The output of the GDN is imposed to the generator, the input dimension of the generator is equal to 6400, and the output dimension of the first layer is 100. The output dimension of the second dense layer is equal to 20,000. The reshape layer converts the shape of the 20,000-dimensional vector to a three- dimensional output to impose to a 2D convolutional layer. Eight two-dimensional output with (50, 50) dimensions are imposed to the first transposed two-dimensional convolution layer.
  - The kernel size in each of the transposed convolutional layers is equal to 4 × 4, and the number of filters in each of them is equal to eight.
  - The size of the strides in the first transposed convolution layer is equal to 2 × 2, in the second transposed convolutional layer, it is equal to 3 × 3 , and in the next two transposed layers, it is equal to 1 × 1. The output of the fourth transposed convolution 2D is imposed to the 2D convolution layer. The kernel size of the 2D convolution layer is considered as being equal to 2 × 2 , and the ize of the strides in this layer is equal to 2 × 2 .
  - The output dimension of this layer is equal to (299, 299), and is imposed to the last reshape layer.
  - The output of the generator is a 299 × 299 dimensional image.
  - The schematic view of the outputs of each layer and the differences in the dimensions of the generator part of the proposed GDN-GAN are illustrated in Figure 6.
  - ![image](https://github.com/user-attachments/assets/e2b346e7-ad34-4451-a9aa-7ed43a04121a)

  - Table 2 gives information about the details of the discriminator part of the proposed network.
    
  - ![image](https://github.com/user-attachments/assets/5a5fc379-5c04-40d7-ad6c-37c10dcbbf46)


## The Adversarial Part
  - The adversarial part of the proposed GAN has three 2D convolution layers with the rectified linear unit as the activation function.
  - The size of the kernel for each of these convolutional layer is considered equal to 4 × 4 , the size of the strides is equal to 2 × 2, and the number of filters for each of them is equal to four.
  - The output of the third 2D convolution is flattened and imposed to a dense layer with an output dimension that is equal to one, to discriminate between fake or real images generated by the generator part of the GAN.

  - Figure 7 illustrates schematic view of dimensions of different layers and it presents a tangible view of the outputs in each phase of the network.
  - ![image](https://github.com/user-attachments/assets/cb31d582-c249-4c97-8299-2c973c1aa29f)
  - Table 3 gives information about the details of the discriminator part of the proposed network.
  - ![image](https://github.com/user-attachments/assets/952c63c0-a7c3-45d8-a946-e1dfbe2655e0)

  
## Training Procedure 
  - Binary cross-entropy is used as a loss function for the GAN part.
  - Discriminator loss is considered as the sum of the loss of the original image and the loss of the generated image.
    - For the loss of the discriminator output of the original image, instead of the ones vector as reference for calculating the cross-entropy between the reference and the original image, 0.9 is used as the coefficient of the ones vector.
    - For the loss of discriminator output of the generated image, the cross-entropy is calculated between the generated image and the zeros vector with dimensions equal to the generated image.
  - Generator loss is considered as the cross-entropy between the generated image and the ones vector with dimensions equal to the generated image.
  - An Adam optimizer with a learning rate equal to 0.0001 is used to train both the generator and discriminator networks.
  - The tuning of different parameters of the proposed GDN-GAN is achieved through a trial–error procedure. Training is performed with the use of different parameters available in Table 4 as a search space. The optimal values for training with good convergence are represented in this table.
  - ![image](https://github.com/user-attachments/assets/e1eafd34-1748-4669-913c-fa4abf2e07f5)


## Saliency Metrics

  -  Ground truth is necessary for calculating these metrics. Another input would be the saliency map. Considering these two inputs and computing these metrics, the degree of the similarity between them would be available.
  -  Similarity (SIM)
  -  Structural similarity (SSIM)
  -  Pearson’s correlation coefficient (CC)
  -  normalized scanpath saliency (NSS)
  -  The shuffled area under the ROC curve (s-AUC)
  -  SSIM interprets the structural similarity index using the mean and standard deviation of pixels of a selected window with fixed size in reconstructed image and the ground truth data, and it would bring a reliable measure of similarity.
  -  The s-AUC uses true positives and false positives according to the pixels of the reconstructed image in the locations of fixations in ground truth data, and is a confident metric of similarity between the two images.


# Results
### Functional Connectivity
  - To illustrate the effect of different visual stimuli on brain activity during the visual process in the brain -> consider the average of the time–domain samples of each EEG channel among all the recordings, in accordance with the particular category of visual stimulation.
  - To show the similarity between brain activity according to different categories -> Representational similarity analysis of signals.
    - This is a good evidence that EEG signals contain visually related information in order to lead a person to the recognition of the surrounding environment 
  - ![image](https://github.com/user-attachments/assets/1cec4106-d510-49bc-93ec-e410fe2189be)

  - The functional connectivity estimation of EEG channels is the first step of the GDN part of the proposed GDN-GAN. Approximating the connectivity matrix according to the specific sparsity level is achieved, and the number of nonzero elements of the corresponding adjacency matrix would decrease to avoid computational complexity.
  - The adjacency matrix would be the sparsely approximated connectivity matrix.
  - ![image](https://github.com/user-attachments/assets/d758d955-c4f8-4e55-9157-80a5d23170be)
    - Figure 10 illustrates the circular connectivity, considering the threshold level for sparsifying the connectivity matrix with the best training convergence result.
    - The circular connectivities for the green (Ch1 − Ch32), yellow (Ch33 − Ch64), red (Ch65 − Ch96), and white (Ch97 − Ch128) electrodes according to Figure 1 are shown seperately.
    - ![image](https://github.com/user-attachments/assets/ade2cc56-ebcc-4b81-b50a-14475f53ee5e)
### Training/Test Accuracy
  - Figure 11a shows the training/test accuracy of GDN part of the proposed GDN-GAN with respect to the number of iterations in this network for the classification of 40 different categories of visual stimuli.
  - ![image](https://github.com/user-attachments/assets/05dfd744-dda8-4d4f-88a9-979c4d04a3ba)
  - Figure 11b shows the training/test loss function variations with respect to the number of iterations in this network for the classification of 40 different categories of visual stimuli.
  - ![image](https://github.com/user-attachments/assets/914a517d-eb54-40bb-89ca-0e57769f0cd5)
  - Figure 12 shows the receiver operating characteristic (ROC) plot for the GDN part of the proposed GDN-GAN and other state-of-the-art methods for classification of the EEG-ImageNet dataset, including region-level stacked bi-directional LSTMs [56], stacked LSTMs [57], and Siamese network [48] . The superiority of the GDN in terms of the area under the ROC can be seen in this figure compared to the other existing methods.
  - ![image](https://github.com/user-attachments/assets/38048a64-25a2-416f-b926-10adda2ecdf0)
  - Furthermore, the performance of the GDN against the above-mentioned state-of-the-art methods in terms of precision, F1-score, and recall metrics is shown in Table 5.
  - ![image](https://github.com/user-attachments/assets/abdb4f39-593e-4540-939c-3c942c644e15)
  - To demonstrate the efficiency of the GDN, we compare the performance of our method with traditional feature-based CNN and MLP. For this purpose, three hidden layers for MLP and CNN with a learning rate of 0.001 have been considered. Maximum, skewness, variance, minimum, mean, and kurtosis have been used as feature vectors for every single channel.
  - According to Figure 13 , feature-based traditional deep networks such as MLP, CNN, and feature-based GDN have a poor performance in the case of classification of the EEG-ImageNet dataset with 40 different categories.
  - ![image](https://github.com/user-attachments/assets/1209a529-da56-4441-9de0-8b6362a276b4)
    - This figure shows the obtained accuracy of feature-based MLP, CNN , and GDN in 50 iterations. This figure illustrates that feature based GDN, MLP, and CNN have shown relatively similar performances, and the training times per epoch for these methods are very high for these traditional methods.
    - This figure confirms the good efficiency of the proposed GDN against the traditional
feature-based deep networks. According to this figure, the proposed network has high
classification accuracy compared to the other networks.
### Confusion Matrix
- A good confirmation to the performance of the proposed method is the confusion matrix shown in Figure 14 .
- ![image](https://github.com/user-attachments/assets/fe707469-d12f-4b85-81e6-91a78cbf9bfe)
- The confusion matrix is an appropriate illustration of the performance of a network on test splits in the case of multi-class classification.
-  Figure 14 shows the confusion matrix of the GDN part of the proposed method. This figure confirms the good performance of the classification part of the GDN-GAN. For the reconstruction phase of the proposed GDN-GAN, the training and evaluation of the proposed GDN-GAN are conducted according to the 10-fold cross-validation.
### Results for each category
- This table illustrates the saliency evaluation metrics according to the proposed method.
- The EEG signals are categorized in first part of the GDN-GAN.
- According to the extracted label, image stimuli is determined
- This image with the extracted feature of the first phase of the proposed method is imposed to the GAN part of the network to map the EEG signal to the saliency map of the image stimuli.
- After training, to test the GAN part, the EEG signals are imposed to the GDN-GAN, and the extracted images are compared to the original ground truth data through different saliency evaluation metrics, and the average of these metrics are reported in this table according to each category.
-  The ground truth data are obtained using open-Salicon. The open-Salicon has been implemented using compiled the Python-compatible Caffe environment. The saliency evaluation metrics of the proposed GDN-GAN for different categories of visual stimuli are reported in detail in Table 6.
-  Furthermore, the overall SSIM, CC, NSS, and s-AUC are represented through computing of the average of the saliency evaluation metrics among all categories.

- ![image](https://github.com/user-attachments/assets/5df7e0e8-c403-465b-b7d9-ed2b124120a4)

- According to this table, the proposed category-level performance of the visual saliency reconstruction method is over 90% except for six categories including Revolver, Running shoe, Lantern, Cellular phone, Golf ball and Mountain tent, in terms of SSIM and s-AUC.
- **SSIM** interprets the structural similarity index using the mean and standard deviation of pixels of a selected window with fixed size in reconstructed image and the ground truth data, and it would bring a reliable measure of similarity.
- The **s-AUC** uses true positives and false positives according to the pixels of the reconstructed image in the locations of fixations in ground truth data, and is a confident metric of similarity between the two images.
- Considering these details, SSIM and s-AUC illustrate the limitations of the proposed GDN- GAN.
- However, considering the detailed values of the four saliency metrics, this table shows that the proposed GDN-GAN is a reliable and efficient method to map the EEG signals to the saliency map of the visual stimuli.
### The Loss Plots
- The trained GDN-GAN for saliency reconstruction is fine-tuned for image construction issues. The loss plots in the result of training the generator and discriminator networks for visual saliency and image reconstruction are represented in Figure 15 for three number of categories.
-
![image](https://github.com/user-attachments/assets/8cabcfdf-c688-4a59-9e49-e303ac255879)

- In addition, the SSIM and CC plots of both visual saliency and image reconstruction per epoch for these categories can be seen in this figure.
- The loss plots corresponding to both the saliency reconstruction and image reconstruction illustrate that the variations in the generator and discriminator loss plots tend to oscillate around one, as saliency evaluation metrics, including SSIM and CC, start to converge ˙These are the behaviors of GANs, and these plots are confirmation of the effectiveness of the proposed reconstruction of the GDN-GAN.
- The results of visual saliency and image reconstruction for all of the 40 categories of image stimuli are illustrated in Figures 16–19.
![image](https://github.com/user-attachments/assets/e4d04d35-96ba-43ad-8817-3b84a7b4e901)
![image](https://github.com/user-attachments/assets/4addb95f-7173-47ac-8942-c4b1f0684f18)
![image](https://github.com/user-attachments/assets/238c02df-d8ea-42de-a6c3-dce319de8029)
![image](https://github.com/user-attachments/assets/139c3847-4255-4d71-8bc8-67f0ac149f04)

- In addition, the ground truth data and the gray-scale versions of the original input image stimuli are shown in these figures.
- The visual evaluations of these figures besides the saliency evaluation metrics confirm the efficiency of the proposed GDN-GAN.
- A comparison of the proposed GDN-GAN with state-of-the-art methods for image saliency extraction is conducted, and the performance metrics are reported in Table 7.
- ![image](https://github.com/user-attachments/assets/33ba9eed-02b4-4c43-8dd5-945093e048fc)
- The results of SalNet [ 43 ], SALICON [42], visual classifier-driven detector, ref. [ 48 ] and neural-driven detector [48] are demonstrated in this table.
- SALICON and SalNet are valuable approaches, considering the image data for saliency map extraction according to the eye-fixation points of the eye tracking process while a subject looking at an image.
- Another valuable approach, the visual classifier-driven detector and the visual neural-driven detector by Pallazo et al. [48] , merges two modalities of EEG signals and image data to extract the image saliency map efficiently.
- Our proposed GDN- GAN is the first method that maps the EEG signals to the corresponding saliency map of the visual stimuli and reconstructs the saliency map and image stimuli. Considering the metrics according to these state-of-the-art methods concerning saliency map extraction in Table 7, this confirms the efficiency of the the proposed GDN-GAN.

# Limiations
1. The ground truth data are generated using the pre-trained Open-Salicon using the image samples corresponding to the EEG-ImageNet database.
  - This point should be considered in future works, and the solution is to use a good eye-tracker device and to record the eye fixation maps at the same time as the EEG recordings.
  - These recorded eye fixation maps should be used as the ground truth data in future works.
2. The two-phase process of saliency reconstruction and three-phase of image reconstruction, considering the functional connectivity based graph representation of the EEG signals imposed as the input to the network.
  - An end-to-end process should be considered as the target deep network to decrease the training phases, eventually reducing the computational complexity, and hence increasing the speed of the network.

# Conclusions
This research would be applicable to BCI projects for helping disabled people to communicate with their surrounding world. 
Neural decoding of the visually provoked EEG signals in BCI will interpret the brain activity of the subject and realize the automatic detection of the stimuli. 
It will pave the way toward mind reading and writing via EEG recordings, and is a preliminary step to help blind people with producing a module to realize vision through the generation of EEG signals corresponding to the visual surrounding environment.


# Chebyshev Graph Convolution

![image](https://github.com/user-attachments/assets/8b902d11-3280-4e92-8c4f-3fc204ec932e)
![image](https://github.com/user-attachments/assets/8a63b16b-92b9-421c-90ac-42131b19df16)
![image](https://github.com/user-attachments/assets/a685f10e-b5a0-43f9-9611-4600740c4fa6)
![image](https://github.com/user-attachments/assets/c0aee884-749d-4aa4-b528-214af26c11f7)
![image](https://github.com/user-attachments/assets/137cc242-c74e-44f1-aca5-3619087deba3)
- Code implementation
  - step 1: Constructing the Graph
  ```
  pip install torch torch-geometric
  import torch
  import torch.nn.functional as F
  from torch_geometric.nn import ChebConv
  from torch_geometric.data import Data
  import numpy as np
  
  # Assume `eeg_epoch` is a NumPy array of shape (num_channels, num_samples)
  # representing an EEG epoch. `num_channels` is the number of EEG channels,
  # and `num_samples` is the number of time points in the epoch.
  
  num_channels, num_samples = eeg_epoch.shape
  
  # Compute the adjacency matrix based on functional connectivity (e.g., correlation)
  adj_matrix = np.corrcoef(eeg_epoch)
  
  # Threshold the adjacency matrix to create a sparse graph
  # (optional step to keep only significant connections)
  threshold = 0.5
  adj_matrix[adj_matrix < threshold] = 0
  
  # Convert the adjacency matrix to edge indices and edge attributes
  edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
  edge_attr = torch.tensor(adj_matrix[adj_matrix.nonzero()], dtype=torch.float)
  
  # Node features can be the EEG data itself, but typically it's some feature vector.
  # Here we use the mean power of the signal in each channel as a simple feature.
  node_features = torch.tensor(np.mean(eeg_epoch, axis=1), dtype=torch.float).view(-1, 1)
  
  # Create the graph data object
  data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

  ```
  - Step2: Chebyshev Graph Convolution Layer
  ```
  class EEGChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(EEGChebNet, self).__init__()
        # Chebyshev convolutional layer
        self.conv1 = ChebConv(in_channels, 16, K=K)
        self.conv2 = ChebConv(16, out_channels, K=K)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply first Chebyshev convolutional layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Apply second Chebyshev convolutional layer
        x = self.conv2(x, edge_index, edge_attr)
        
        return x

  ```
  - Step 3:
  ```
  # Initialize the network
  K = 3  # Order of Chebyshev polynomial (typically small, e.g., 3 or 5)
  model = EEGChebNet(in_channels=1, out_channels=4, K=K)  # Adjust out_channels as needed
  
  # Forward pass through the network
  output = model(data)
  
  print(output)
  ```

# GDN Tutorial

![image](https://github.com/user-attachments/assets/a6673f56-aa8f-470c-8fc4-d4eba285ae7b)
![image](https://github.com/user-attachments/assets/16e08882-5857-49c1-ae23-f698a07a6971)

# Saliency Metrics Tutorial

![image](https://github.com/user-attachments/assets/69338e24-6c76-446e-b401-55ed8f6e45ff)
![image](https://github.com/user-attachments/assets/5ba41f22-68d4-402a-b2be-7de378454f1f)
![image](https://github.com/user-attachments/assets/82e4d167-f9f0-42b3-beca-db9305dfcd0a)
![image](https://github.com/user-attachments/assets/46ae038f-db77-461c-8e3d-f4e713e917f2)
