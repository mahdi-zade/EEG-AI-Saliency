# Dataset
The [MindBigData](https://mindbigdata.com/opendb/) has been recorded by David Vivancos with the Emotiv EPOC device. This dataset has been gathered using a 14-channel cap. The placement of the EEG
channel electrodes is shown in Figure 1, and the channel names considered in the recording
procedure are A.F.3, F.7, F.3, FC.5, T.7, P7, O.1 in the left hemisphere and A.F.4, F.8, F.4, F.C.6, T.8, P8, O.2 in the right hemisphere as shown in dark blue in this figure. 
- ![image](https://github.com/user-attachments/assets/f72647c4-9422-4121-bb32-1293ebafecdc)

The channel names with details are described in Table 1.
- ![image](https://github.com/user-attachments/assets/a81a8094-87c0-44c1-961a-e94daa3fd544)

A total of 9120 brain signals of 2 s captured at a theoretical sampling rate of about
128 samples per second or 128 Hz are selected as MindBig dataset used in this paper. Total
number of samples in each channel used for processing is 250. The brain signals were
captured while a single digit from 0 to 9 corresponding to digits of the Modified National
Institute of Standards and Technology (MNIST) dataset in Figure 2 has been shown for 2 s.
The numbers have been represented on a 65-inch TV screen in a white font over a total
black background. The appearance of digits was random, with a black screen between
them. The number of 9120 EEG records corresponding to 912 signals of each category are
considered for classification and salient arithmetic data extraction.
- ![image](https://github.com/user-attachments/assets/cf5e17c5-79f1-4e5e-ad95-704d59eaa456)

# CNN
- The one-dimensional convolutional layers in the first part of the proposed network classify the EEG signals related to 10 different numbers of MNIST images with arithmetic content.
- After the preprocessing stage of the recorded EEG signals in response to the visual image with arithmetic content, we have several deep layers to extract the output vector of the first part of the proposed network to be applied to the next generator adversarial network.
- Figure 3 represents the schematic of the CNN fragment to classify the input EEG signal into the correct category of the MNIST dataset.
- ![image](https://github.com/user-attachments/assets/7ea1d32f-03d8-4ee5-936e-6f7b31685a6d)
- The visual stimulation related to the MNIST dataset appears on an LCD to a human volunteer, corresponding to the considered timing of occurrence for each image and time-lapse between sequential images.
- The EEG signals are recorded during the experiment.
- Normalizing the EEG time samples is performed considering each EEG channel’s mean and standard deviation.
- The pre-processed EEG signals are applied as input to the proposed one-dimensional CNN network.
- The structure of the CNN fragment of the network consists of three convolutional layers, as illustrated in Figure 3.
- The rectified linear unit is selected for the activation function in each layer.
- After each convolutional layer, a dropout and batch normalization are considered to prevent overfitting.
- Flattening of the output of the third convolutional layer is achieved.
- It is imposed to a linear layer, and the output vector of a dense, full-connected layer is passed through a log_softmax classifier layer to classify the input EEG signal.
- **The output of the linear layer is a vector with 2500 elements, and it is the vector applieded as input to the next deep network of the proposed CNN-GAN to extract and obtain the MNIST images used as the stimulation.
- The details of the CNN network of the proposed architecture are explained in Table 2
- ![image](https://github.com/user-attachments/assets/bc566f8b-2b3c-4cc1-b983-9c4cc43fd12e)


# GAN

- The GAN consists of generator and discriminative networks in order to map the one-dimensional extracted feature vector of the EEG signal in the first part to the two-dimensional image array.
- The main layers of GAN in the proposed technique are two-dimensional convolutional blocks to reconstruct the salient images.
- ![image](https://github.com/user-attachments/assets/34e13185-addf-4249-9144-29a4c255f158)
- The salient images are created using the SALICON approach to apply as the ground reference data of the GAN network.
- The generative adversarial network is trained, and the network weights will be determined. After this stage of the procedure, tuning the weights of the trained network will be performed and transfer learning will be employed to reconstruct the original visual image stimulation.
- The architectural details of the generator and adversarial networks can be seen in Figure 5.
- ![image](https://github.com/user-attachments/assets/0da07c7d-9f3d-4b3a-8030-12c6d4e85221)
- Details of the image: The flattened vector with 2500 elements is passed through a dense layer of the generator network and four transposed two-dimensional convolution layers.
- Also, an additional two-dimensional convolutional layer is needed to fit the output image dimension to the desired dimension.


- The adversarial consists of three sequential layers of two-dimensional convolutional layers. The output of these layers is imposed to the dropout layer, and then the flattened output vector passes a fully connected layer to judge about fake or real data.
- The structural and dimensional details of layers in the generator are represented in Figure 6 and Table 3.
- ![image](https://github.com/user-attachments/assets/90421f91-2e0f-421b-809f-9a4a8a687cd1)

- The input vector dimension is equal to 2500, and after passing through two dense layers, the output vector dimension is equal to 20,000. The reshaped vector of dense layer output is applied in the first transposed convolution layer.
- The number of kernels in transposed convolutional layers is considered equal to four to have four two-dimensional outcomes in each layer.
- Considering different kernels and strides in five transposed convolution layers according to Table 3, the dimensions of two-dimensional outputs are equal to 50 × 50, 100 × 100, and 300 × 300, as illustrated in this table.

# Evaluation Metrics
The saliency feature map and the ground truth feature vector are two necessary inputs
for calculating the saliency evaluation metrics. The level of similarity can be represented by
considering these metrics.
- Sensitivity, accuracy, precision, and recall based on true positive (TP), true negative (TN), false positive (FP), and
false negative (FN)
- Similarity metric (SIM)
- Structural similarity (SSIM)
- Correlation coefficient (CC)
- The covariance between FM and SM

# Training
- The training procedure is accomplished through cross-validation to adjust the network weights of the proposed CNN to the MindBig dataset.
- The trained parameters of this part are transferred to the proposed CNN-GAN to extract the salient images of visual stimulation from EEG recordings.
- Cross-entropy is utilized as a loss function for the training phase of CNN, and binary cross-entropy is employed for the training phase of CNN-GAN.
- Different parameters have been used through a trial–error technique to find the optimal values of the proposed architecture.
- Table 5 represents the values as search scope for the optimizer, cost function, and learning rate for the CNN and generative adversarial parts.
- The corresponding optimal values are obtained with trial and error and illustrated in this table.
- ![image](https://github.com/user-attachments/assets/86b984be-6afd-4456-b90e-1c254c3f5552)



# Results

#### The train and test accuracy plots of the proposed CNN
- Classifies the MindBig dataset into 10 categories of visually evoked brain signals corresponding to the numbers between zero and nine Figure 9.
- ![image](https://github.com/user-attachments/assets/216a53b9-27a5-4325-9fbf-a5544047044a)

- The train and test loss plots of the proposed convolutional network are illustrated in Figure 10.
- ![image](https://github.com/user-attachments/assets/d3f80c45-93f4-47d0-b957-c8429485792a)

##### Comparison with other methods
- ![image](https://github.com/user-attachments/assets/06c111da-1dbd-4649-8ed0-7ab378940c85)
- ![image](https://github.com/user-attachments/assets/4cf2a636-4782-4559-ab4e-3fc334e14e55)
- ![image](https://github.com/user-attachments/assets/98848e57-c3c1-4bfc-9983-32fe27a1f53f)
- Another way to assess the performance of the proposed architecture is the representation of the confusion matrix.
- Figure 15 exhibits the corresponding matrix of the proposed CNN to classify the MindBig dataset.
- This matrix confirms the efficiency of the proposed CNN.
- ![image](https://github.com/user-attachments/assets/c39f6a7e-b745-4113-8a4e-5e9408d1036a)


#### Generating EEG Signals using GAN 

- The MindBig dataset consists of 9120 14-channel EEG signals.
- We can generate a set of signals through training a generative adversarial network (GAN) and add the generated signals to the base dataset to evaluate the performance of the proposed CNN.
- We generate 50 sets of 10 14-channel EEG signals according to different categories and add these 500 generated signals to the MindBigData.
- The generator part of the GAN consists of three transposed convolution 2-D layers, and the discriminator part of the GAN includes three convolution 2-D layers.
- The details of layers are presented in Tables 5 and 6.
- ![image](https://github.com/user-attachments/assets/254a2f90-30ff-4b34-9162-aae48058dd8d)
- ![image](https://github.com/user-attachments/assets/3634557a-7f99-4495-a1f7-a264788c8190)
- The training of GAN is performed with 9000 iterations.
- The test accuracy of the proposed CNN with the new dataset after 10-fold cross-validation is equal to 90.3%.
- The accuracy of the network with the pre-trained weights is performed considering the gen- erated signals, and the obtained test accuracy is equal to 86.9%.
- Figure 16 confirms the proposed network's efficiency for classifying the new MindBigData with 9620 14-channel EEG recordings.
- ![image](https://github.com/user-attachments/assets/f4ffc2ce-2032-418e-8d2a-f03951157270)

- The details of the layers related to the generative and discriminator subnets are presented in Tables 7 and 8, respectively.
- ![image](https://github.com/user-attachments/assets/e4976100-5cc3-46da-8b7c-374d02f6bcd4)

#### Evaluation of reconstructed salient image
- The evaluation of the proposed CNN-GAN for salient image extraction is accomplished in a 10-fold cross-validation considering the SSIM and CC for each image category of the MNIST dataset.
- The classification of the EEG database is performed in the first CNN part of the network, and the extracted feature vector is applied as the input to the next GAN deep network to extract the salient image corresponding to the visual stimulation.
- The results of SSIM and CC are represented in Table 9 and confirm the good performance of the proposed CNN-GAN for reconstructing the salient visual stimulation.
- ![image](https://github.com/user-attachments/assets/1175e598-243d-45b2-af41-57bf040b8ad6)


#### Full Image reconstruction

- The weights of the trained CNN-GAN for salient image extraction from the brain activity are transferred to the network in order to reconstruct the original image.
- The initialization of the parameters is performed in the transfer learning procedure, and the adjustment of the new weights to reconstruct the visual stimulation images is accomplished by tracking the loss function of the generator and discriminator networks.
- The cross-entropy trend curves corresponding to loss function in salient image and original image extraction are illustrated in Figure 17.
- ![image](https://github.com/user-attachments/assets/ddabdf83-abde-4d31-847c-35e80be86b7c)

- Furthermore, the plots of tracking the CC and SSIM metrics according to each iteration are represented in this figure considering four categories in the MNIST dataset.
- The results of the extracted salient image and reconstructed original image according to four visual stimulation groups are represented in Figure 18.
- ![image](https://github.com/user-attachments/assets/a01ec496-dff2-4c4a-9af1-7b33cc750ccb)

- Furthermore, the ground truth image and the actual visual stimulation image are described in this figure.
- The visual assessment and the evaluation of the SSIM and CC metrics validate the efficiency and good performance of the proposed CNN-GAN framework.
- Table 10 compares the performance of the proposed CNN-GAN against other valuable state-of-the-art method methods of SALICON [38], SalNet [39], visual classifier-driven detector [44], neural-driven detector [44], and GNN-based deep network [45] for saliency reconstruction.
- ![image](https://github.com/user-attachments/assets/fc80474c-460e-48a4-a250-6bbac4e382bd)
- This table confirms the efficiency of the proposed CNN-GAN method.


# Limitations
- One of the restrictions of the proposed method to be overcome in future works is constructing the reference dataset for salient image extraction.
- This article’s reference data for visual saliency is gathered by implementing the SALICON technique in the CAFFE environment compiled to be compatible with the Python programming language.
- This would be considered in future works to have salient data using an eye-tracker for tracking the pupil position to identify the visual salient part in the images.
- Another recommendation to be considered in future works is more complicated arithmetic content for visual stimulation, and EEG records could be analyzed in these complex situations.
- Channel selection is another recommendation to be explored in future works.
- Studying the effects of different EEG channels in classification and salient arithmetic content extraction would be beneficial. The channels with the most discriminative data could be
diagnosed through the experiment.
- The application of this research in BCI projects must be addressed. The implementation of the proposed method in this article in the real world would be helpful for disabled or blind subjects to have better interaction with the surrounding environment.







# Evaluation Metrics Tutorial


The most used metrics for classification are described in (3) as sensitivity, accuracy,
precision, and recall based on true positive (TP), true negative (TN), false positive (FP), and
false negative (FN)
- ![image](https://github.com/user-attachments/assets/899eebb7-bbc5-4227-b794-6141002a1a39)

Three important metrics for evaluating saliency are described as follows.
Similarity metric (SIM) is used to measure similarity between distributions. Nor-
malization of the input signal vectors is performed, and the sum of minimum values at
each pixel results as S.I.M. The saliency map is shown with S.M and the fixation map is
represented with F.M:
- ![image](https://github.com/user-attachments/assets/9ce489fe-4b5c-4919-915a-a0a097eff41f)

In (5), pixel locations are represented with j. The value of SIM is equal to one for inputs
with identical distributions, while this metric would be zero if there is no similarity and
overlap between distributions.

Another saliency evaluation metric is structural similarity (SS.I.M), calculated using
different windows of an image. The SSIM is computed considering two sampling
windows, m and n, with size L × L:
![image](https://github.com/user-attachments/assets/b21507a5-9f06-4aa9-babb-46e865295565)

A metric for assessing the affine connectedness of distributions is Pearson’s correlation
coefficient (CC) [52]. CC can be computed as in Equation (7).
![image](https://github.com/user-attachments/assets/99106d57-06b3-4ba2-804b-0f1451a1e544)

The covariance between FM and SM in (7) is represented with σ(SM, FM). This metric
is unaffected by linear transformations. This evaluation metric would treat false negatives
and false positives equally and, therefore, is a symmetric function. The similar magnitudes
of the saliency map and the reference ground truth would result in high positive values
of CC.

# CNN Tutorial
Each deep convolutional network is composed of several layers. The pooling layer is
an essential layer in CNN, which minimizes the spatial size of feature vector maps obtained
from the convolutional layer. This layer has no training parameter and performs a simple
sampling. The most famous pooling layers are called average pooling and max pooling.
For example, for maximum integration, a predefined window is considered that moves
over the image to select the maximum value and ignore the rest of the numbers. The size
of the filter and the size of the stride step in this layer are considered proportional to the
optimal size for mapping the obtained feature of each layer.
The fully connected layer forms the final layer of CNN networks, which is used to
classify the extracted feature maps. This layer is similarly present in multilayer perceptron
(MLP) networks. After displaying the feature vectors obtained from convolutional layers,
weight vector coefficients are assigned in this layer. The output corresponding to the
number of classes available for classification can be achieved.
The following describes some other commonly used layers in CNN networks, includ-
ing the random elimination layer and the batch normalizer layer. The use of the dropout
layer in CNN networks strongly prevents the phenomenon of data overfitting in the train-
ing process. The function of this layer is to omit some neurons during training randomly.
To optimize the coefficients, these randomly selected neurons are not considered during
the learning process. Mathematically, neurons are discarded with probability (p-1), and
other neurons are retained with probability (p).
The normalization layer is used to normalize the data inside the network. By perform-
ing various calculations on data, the distribution of data will change. The batch normalizer
layer increases the training speed of the network by reducing the internal covariance of
data distribution and accelerates the convergence process. The performance of this layer
will be based on the calculation of the average and variance of data according to (1).
![image](https://github.com/user-attachments/assets/d18503f8-1b6e-4d80-afbb-a4bc891e70cd)

# GAN Tutorial
The generative adversarial networks (GANs) [7] include generator and discriminator
networks. The generative model G consists of some layers to fit a random vector y with
probability distribution P(y) into a desired data distribution. The discriminative part, named
D, compares data from the expected distribution and data obtained from the generator
part. These two networks are trained simultaneously, and the training will continue to
see no improvement in network optimization. The cost function of GAN can be described
as follows:
![image](https://github.com/user-attachments/assets/605d9404-8ce8-40cd-aeaf-a93e82e87c15)
![image](https://github.com/user-attachments/assets/10f00c41-b647-474a-a31e-3344bac35bff)
