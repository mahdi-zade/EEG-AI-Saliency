# EEG-AI-Saliency


## Abstract



## Introduction
Understanding how the brain processes external stimuli is crucial for advancing many fields. namely assistive tecnologies, cognitive neuroscience, education, sports, Consumer and Lifestyle Technologies, Marketing and User Experience. 

Saliency detection, a fundamental aspect of human visual perception, allows us to efficiently process scenes by focusing on the most important elements, thereby optimizing cognitive resources.
Visual saliency is the identification of key elements in a scene that stand out due to their visual distinctiveness or task relevance.

The representation of visual stimuli in the brain relates to important points of the picture. To illustrate the priority of a location in a visual image to represent in the brain, and to identify them efficiently, the concept of a saliency map was first proposed by Koch and Ulman in 1985 [18].

Traditional approaches of saliency detection [2] attempt to identify salient areas in a biologically-plausible way, through the analysis of hand crafted multi-scale color/intensity/orientation maps.

In a number of early works on salient region detection [20 – 22 ], saliency was considered as being unique, and was frequently calculated as center–surround contrast for every pixel. 

In 2005, Hu et al. [23 ] used generalized principal component analysis (GPCA) [24 ] to compute salient regions. GPCA has been used to estimate the linear subspaces of  the mapped image without segmenting the image, and salient regions have been determined by considering the geometric properties and feature contrast of regions. 

Rosin [ 25 ] pro- posed an approach for salient object detection, which has required very simple operations
for each pixel, such as moment preserving binarization, edge detection, and threshold decomposition. 

Valenti et al. [26 ] proposed an isophote-based framework where isocenter clustering, color boosting, and curvedness have been used for the estimation of the saliency map.

In addition, some supervised learning-based models for saliency detection were proposed, such as support vector machine in 2010 with Zhong et al. [27 ], regression in 2016 with Zhou et al., and neural networks with Duan in 2016 [28]. Some of the saliency detection methods are based on models developed for simulating the visual attention processes. Visual attention is a selective procedure that occurs for understanding the the visual input to the brain from the surrounding environment. Neisser, in 1967 [ 29 ], suggested that bottom-up and top-down processes occur in the brain during the time of the processing objects of a visual scene.

A number of researchers have made efforts to improve the performance of the bottom-up-based saliency models. In 2013, Zhang and Sclaroff measured the contour information of regions using a set of Boolean maps to segment the salient objects from the background, and the efficiency of the model was demonstrated by five sets of eye tracking databases [30]. 

In 2015, Mauthner et al. proposed an estimation of the joint distribution of motion and color features based on Gestalt theory,in which the local and global foreground saliency likelihoods have been described with an
encoding vector, and these individual likelihoods generated the final saliency map [31]. 

Top-down saliency-based models, as in the work of Xu et al. in 2014 [32],have been conducted through contextual guidance and pre-defining of the discriminant features and allocating learned weights for different features, as performed by Zhao and Koch in 2011 [ 33 ], and in 2017, Yang [34 ] adapted feature space in a supervised manner to obtain the saliency output.

Although real-time saliency detection with hand-crafted features has good performance, it does not work well in challenging scenarios to capture salient objects. One of the proposed solutions to these challenges is using neural networks [35,36] . Newer approaches [3], [4], [5] directly train CNNs using saliency maps as target signals. However, the accuracy of automated saliency detection approaches is still far from the human level.
One of the most popular networks in machine learning are convolutional neural networks (CNNs) [35 ], and they have been implemented to solve a number of vision problems such as edge detection, semantic segmentation [37 ], and object recognition [38 ]. 

Recently, in the work by Shengfeng He et al. and Ghanbin Li et al. [39 , 40 ], the effectiveness of CNNs has been shown when applied to salient object detection. 

A series of techniques has been proposed to learn saliency representations from large amounts of data by exploiting the different architectures of CNNs. 

Some of the models proposed for saliency detection via neural networks use multilayer perceptrons (MLPs). In these models, the input image is usually oversegmented into small regions and feature extraction is performed using a CNN. The extracted features are fed to an MLP to determine the saliency value of each small region. 

The saliency problem in [ 39 ] has been solved using the one-dimensional convolution-based methods by He et al. Li and Yu [ 40] have utilized a pre-trained CNN as a feature extractor, such that the input image has been decomposed into a series of non-overlapping regions and a CNN with three different-scale inputs has been proposed to extract features from the decomposed regions. Advanced features at different scales have been captured using three subnetworks of the proposed CNN, and have been concatenated to feed into a small MLP with only two fully connected layers. These dense layers act as a regressor to output a distribution over binary saliency labels. 

Two recently proposed deep learning-based saliency models are salicon [ 41 ,42 ] and salnet [ 43 ]. Like other saliency detection methods, the purpose of the salicon is to realize and to predict visual saliency. This model has used the coefficients of pre-trained AlexNet, VGG-16, and GoogleNet. The last layer of the proposed salicon is a convolutional layer that is used to extract the salient points. The initial parameters have been determined
using the pre-trained network based on ImageNet dataset, and the back propagation has been used to optimize the evaluation criterion, in spite of previous approaches that used support vector machine. The training process in salnet has been achieved using the Euclidean distance between the mapped predicted salient points and the ground truth pixels. A shallow and a deep network have been presented. The shallow net consists of
three convolutional layers and two fully connected layers with trained weights. ReLU is used as the activation function of each layers of shallow net. The deep network consists of 10 layers and 25.8 million parameters.

Deep Supervised Salient Object Detection (SSOD) excessively relies on large-scale annotated pixel-level labels which consume intensive labour acquiring high quality labels. 

Lastest research attempts to merge biological priors with the representational power of deep architectures. 
Because the recent discoveries show evidence that brain representations in the visual pathway appear to
be highly correlated to activation patterns within neural network layers [6], [7].

Electroencephalography (EEG), a widely used non-invasive brain recording technique, has been instrumental in studies exploring brain activity across various contexts, including attention, memory, motor control, and vision. 

In recent years, some efforts have been made to understand the connection between the visual saliency content and the brain activity. 

In 2018, Zhen Liang et al. [44 ] presented a model to study this connection and extracted sets of efficient features of EEG signals to map to the visual salient related features of the video stimuli. The model has used the work of Tavakoli et al. in 2017 [ 45]. The reconstruction of the features of the salient visual points based on the features of the EEG signal has been performed with good accuracy in [44], and prediction of the temporal distribution of salient visual points has been done using EEG signals recorded in a real environment. 

In another study [ 46 ], the identification of the objects in images recorded by robots was the purpose of the study, and a method based on P300 wave was applied to identify the objects. The significant challenge for extracting the objects of interest in navigating the robots is how to use a machine to extract the objects of
interest for humans. The combination of a P300-based BCI and a Fuzzy color extractor has been applied to identify the region of interest. 

Humbeeck et al. [ 47] have presented a model for calculating the importance of the salient points for the fixation positions. Brain function related to the extracted model has been studied using the eye-tracker and recording the EEG signal. An evaluation of the connection between the importance of salient points and
the amplitude of the EEG signal has been done via this modeling. 


A multimodal learning of EEG and image modalities has been performed in [48 ] to achieve a Siamese network for
image saliency detection. The idea of the work in [ 48] is the training of a common space of brain signal and image input stimuli by maximizing a compatibility function between these two embeddings of each modality. The estimation of saliency is achieved by masking the image with different scales of image patch and computing the corresponding variation in compatibility. This process is performed at multiple image scales, and results in a saliency map of the image.

Integrating deep learning techniques with EEG signal analysis has shown promise in enhancing our understanding of how visual stimuli influence brain activity. For instance, deep networks have been employed to classify EEG responses to visual inputs and even reconstruct images from brain signals, highlighting the potential of this interdisciplinary approach to bridge the gap between neural activity and visual perception.

In such precondition, deep Unsupervised Salient Object Detection (USOD) draws public attention. Under the framework of the existing deep USOD methods, they mostly generate pseudo labels by fusing several hand-crafted detectors’ results. On top of that, a Fully Convolutional Network (FCN) will be trained to detect salient regions separately. While the existing USOD methods have achieved some progress, there are still challenges for them towards satisfactory performance on the complex scene, including (1) poor object wholeness owing to neglecting the hierarchy of those salient regions; (2) unsatisfactory pseudo labels causing by unprimitive fusion of hand-crafted results. To address these issues, in this paper, we introduce the property of part-whole relations endowed by a Belief Capsule Network (BCNet) for deep USOD, which is achieved by a multi-stream capsule routing strategy with a belief score for each stream within the CapsNets architecture. To train BCNet well, we generate high-quality pseudo labels from multiple hand-crafted detectors by developing a consistency-aware fusion strategy. Concretely, a weeding out criterion is first defined to filter out unreliable training samples based on the inter-method consistency among four hand-crafted saliency maps. In the following, a dynamic fusion mechanism is designed to generate high-quality pseudo labels from the remaining samples for BCNet training. Experiments on five public datasets illustrate the superiority of the proposed method. 

In 2010, Ghebreab et al. [13] investigated the recorded EEG signals in response to natural visual stimulation, and the prediction of visual inputs was realized using EEG responses. A better accuracy was achieved in comparison to a similar work by Kay et al. in 2008 [14]. These studies have had the potential to reveal the effects of visual features such as color [15 ], orientation [16 ], and position [17 ] on the brain signals of the visual cortex.

The representation of visual stimuli in the brain relates to important points of the picture. To illustrate the priority of a location in a visual image to represent in the brain, and to identify them efficiently, the concept of a saliency map was first proposed by Koch and Ulman in 1985 [ 18 ]. Followed by the concept introduced by Koch and Ullman, Itti et al. in 1998 introduced a computational model corresponding to the understanding of the saliency map [ 19 ]. Following the work of Itti et al. in 1998 [19 ], detecting rarity, distinctiveness, and uniqueness in a scene is compulsory for salient object detection. 

Based on the proposed model by Itti [19], many models have been developed for predicting image saliency.
Realizing how the salient region affects the brain signal is of great importance to understanding how the visual system works. 

Although some works have been made to explore the relationship between the brain activity through recorded EEG signals and the salient regions of the visual stimuli, the mapping of the EEG signals to image saliency has not been realized. 


## Methods

There are many ways to classify various methods used for visual saliency detection.

The most recent review paper in the field has classified the mothods into two broad categories:
- Studies have shown a strong correlation between brain representations in the visual pathway and activation patterns within deep neural networks.
- This has led to research into "brain-guided" saliency detection, using neural signals like EEGs as an additional input or supervisory signal for deep learning models.

![image](https://github.com/user-attachments/assets/9a204af3-b439-452b-87ac-53710bbe8899)


### Conventional Methods for Saliency Detection
Traditional approaches to saliency detection aim to identify salient areas in images by analyzing hand-crafted features and designing algorithms that simulate human perception, often inspired by biological processes in the human visual system. These methods often analyzed low-level visual cues like color, intensity, and orientation to create saliency maps, highlighting regions that stand out from their surroundings

#### Limitations 
Traditional approaches for visual saliency detection have limitations, particularly when compared to the capabilities of the human visual system and newer deep learning methods.

- <details>
  <summary>Traditional methods often rely on hand-crafted feature maps based on assumptions about the human visual system.</summary>
  These hand-crafted features might not capture all the complexities of visual attention and saliency. For instance, early models inspired by primate visual systems used multi-scale image features to build saliency maps, even integrating biologically-inspired mechanisms to simulate attention shifts.
</details>

- <details>
  <summary>The accuracy of traditional automated saliency detection methods lags behind that of humans.</summary>
  While they provide a foundation for understanding and implementing saliency detection, their limitations become evident in complex scenarios where capturing salient objects is challenging.
</details>

- <details>
  <summary>The success of deep learning in visual tasks has led to the development of more robust saliency detection networks.</summary>
  Deep learning models can learn more complex and intricate representations from vast amounts of data, potentially surpassing the limitations of hand-crafted features. This has shifted the field towards utilizing the power of deep learning for improved saliency detection.  For example, deep learning methods like SALICON and SalNet utilize pre-trained networks and vast datasets to achieve better saliency prediction than earlier approaches.
</details>

- <details>
  <summary>The emergence of deep learning in visual saliency detection, however, doesn't completely diminish the value of traditional methods.</summary>
  These earlier approaches still provide valuable insights into the human visual system and offer a basis for comparison and further development. The field continues to advance by exploring new architectures and leveraging the strengths of both traditional and deep learning approaches.
</details>



### Deep Learning Methods for Saliency Detection

#### Advantages

- <details>
  <summary>Superior Accuracy and Performance:</summary>
  Deep learning models, especially convolutional neural networks (CNNs), have consistently demonstrated superior accuracy in visual tasks like saliency prediction compared to traditional methods. This performance gap stems from deep learning's ability to learn complex representations from vast amounts of data, moving beyond the constraints of hand-crafted features.

</details>

- <details>
  <summary>Learning Complex Representations:</summary>
  Deep learning methods can automatically learn intricate and hierarchical representations of features from raw data, without the need for manual feature engineering. This data-driven approach allows them to capture subtle patterns and nuances in visual data that might be overlooked by hand-crafted features.

</details>

- <details>
  <summary>Adaptability and Generalizability:</summary>
  Once trained on a large dataset, deep learning models can be fine-tuned and adapted for different tasks and datasets.  For instance, a model pre-trained on ImageNet, a vast image database, can be effectively fine-tuned for saliency prediction on a different dataset.

</details>

- <details>
  <summary>Handling Complexity:</summary>
  Deep learning models excel in handling the complexity of real-world visual scenes. They can discern salient objects from cluttered backgrounds, a task that often poses challenges for traditional methods reliant on simpler feature representations.

</details>

- <details>
  <summary>End-to-End Learning:</summary>
  Deep learning enables end-to-end learning, where the model learns to map directly from raw input (e.g., images) to the desired output (e.g., saliency maps).  This eliminates the need for separate feature extraction and selection steps that are typical of traditional methods.
</details>

### Limitations

- <details>
  <summary>Dependence on Large Datasets</summary>
  Deep learning models, particularly CNNs, are notorious for their hunger for large labeled datasets. These models thrive on identifying patterns and features within vast amounts of data.  Collecting, annotating, and curating such datasets, especially for a nuanced task like saliency detection where ground truth data often come from eye-tracking studies, can be prohibitively expensive, time-consuming, and challenging to scale.

</details>

- <details>
  <summary>Generalization Challenges</summary>
  The impressive performance of deep learning models often stems from their ability to memorize intricate patterns within the training data. However, this can sometimes lead to overfitting, where the model excels on the training data but struggles to generalize well to unseen data or different contexts.  This is a particular concern with saliency detection, where the models need to be robust and adaptable to diverse visual scenes and variations in human attention.

</details>

- <details>
  <summary>Lack of Interpretability (Black Box Problem)</summary>
  Deep learning models, with their layered architectures and millions of parameters, are often regarded as "black boxes."  While they excel at generating accurate predictions, understanding how and why a model arrived at a specific saliency map remains a challenge. This lack of interpretability can limit our ability to trust, debug, and refine these models effectively, especially in critical applications where understanding the decision-making process is crucial.

</details>

- <details>
  <summary>Computational Demands</summary>
  Training complex deep learning models, especially with large datasets, demands significant computational resources. This can pose a barrier for researchers and developers without access to powerful hardware (GPUs) and can increase energy consumption.

</details>

- <details>
  <summary>Difficulty in Incorporating Biological Priors</summary>
  While the sources demonstrate attempts to bridge deep learning with insights from neuroscience, effectively incorporating biological priors (knowledge from human brain studies) into deep learning models for saliency detection remains a challenge. Designing architectures and training paradigms that genuinely reflect the complexities of human visual attention is an area of ongoing research.

</details>

- <details>
  <summary>Need for End-to-End Solutions</summary>
  Many current deep learning approaches for saliency detection, including those mentioned in the sources, often involve multi-stage processes. For instance,  in pre-trained models like AlexNet and GoogleNet are adapted for saliency prediction. Similarly, the proposed GDN-GAN in  involves a two-phase process of classification using a CNN followed by saliency map generation with a GAN. Achieving more streamlined end-to-end solutions where the model seamlessly learns to predict saliency directly from raw EEG signals without intermediary steps is a desirable goal.

</details>

- <details>
  <summary>Limited Availability of High-Quality Ground Truth Data</summary>
  The accuracy of saliency detection models is inherently tied to the quality and representativeness of the ground truth data they are trained on.  As mentioned earlier, obtaining accurate saliency maps usually relies on eye-tracking studies, which can be resource-intensive and prone to noise.  Developing more robust methods for acquiring and validating ground truth data is crucial for further advancing deep learning models in this domain.
</details>


#### Current Deep learning Approaches to saliency detection
Two recently proposed deep learning-based saliency models are salicon [41,42] and
salnet [43]. Like other saliency detection methods, the purpose of the salicon is to realize
and to predict visual saliency. This model has used the coefficients of pre-trained AlexNet,
VGG-16, and GoogleNet. The last layer of the proposed salicon is a convolutional layer
that is used to extract the salient points. The initial parameters have been determined
using the pre-trained network based on ImageNet dataset, and the back propagation
has been used to optimize the evaluation criterion, in spite of previous approaches that
used support vector machine. The training process in salnet has been achieved using the
Euclidean distance between the mapped predicted salient points and the ground truth
pixels. A shallow and a deep network have been presented. The shallow net consists of
three convolutional layers and two fully connected layers with trained weights. ReLU is
used as the activation function of each layers of shallow net. The deep network consists of
10 layers and 25.8 million parameters.

- <details>
  <summary>Multi-scale Patch-Based Methods</summary>
  These methods utilize CNNs to process image patches at multiple scales, capturing both global and local context for saliency prediction.
    *   The sources provide examples like SuperCNN, SALICON, and SalNet that employ this approach. 
    *   Typically, these models involve oversegmenting the input image into small regions or patches.
    *   Features are extracted from these patches using CNNs, often pre-trained on large image datasets like ImageNet for better performance.
    *   The extracted features are then fed into an MLP (Multilayer Perceptron) to determine the saliency score for each region or patch.
    *   For instance, in, Li and Yu propose a CNN that takes image regions at three different scales as input. Three subnetworks within the CNN capture features at these different scales, and the concatenated features are passed to a small MLP for saliency prediction.
 
</details>

- <details>
  <summary>End-to-End Saliency Map Estimation</summary>
  Unlike patch-based methods, some CNN-based models are designed to directly predict a saliency map for the entire input image in an end-to-end manner. These models are typically fully convolutional networks.
    *   One example is the approach by Pan et al., where a fully convolutional CNN, partly utilizing pre-trained layers, is used for saliency prediction.
    *   Another example is from Huang et al., who present a fully convolutional architecture that processes images at two different scales, leveraging deep neural networks trained for object recognition.

</details>

- <details>
  <summary>Leveraging Pre-trained Networks</summary>
   The sources emphasize the effectiveness of utilizing CNNs pre-trained for object recognition on large-scale datasets like ImageNet. This transfer learning approach allows saliency detection models to benefit from the rich feature representations learned by these pre-trained networks.
    *   Kummerer et al. demonstrate the success of this approach by employing networks pre-trained for object recognition to achieve state-of-the-art performance in fixation prediction.
    *   Similarly, Murabito et al. propose a saliency detection method driven by visual classification, highlighting the synergy between object recognition and saliency detection. 
</details>

- <details>
  <summary>Multimodal Learning with Neural Signals</summary>
  This emerging approach combines the power of deep learning with insights from neuroscience by incorporating brain activity data, like EEG signals, to guide saliency detection.
    *   The core idea is to learn a shared embedding space between visual content (images) and corresponding brain signals (EEGs). 
    *   This approach aims to capture the neural correlates of visual saliency by training models that can find correspondences between visual features and brain activity patterns.
    *   One specific method involves training two encoders, one for images and one for EEGs, to maximize the compatibility or similarity between the learned embeddings for corresponding image-EEG pairs. 
    *   Saliency is then estimated by analyzing how this compatibility score changes when different image patches are masked or suppressed at multiple scales.
    *   The sources showcase the potential of this brain-guided approach, demonstrating its ability to achieve state-of-the-art saliency detection performance without relying on direct supervision from ground-truth saliency maps.
</details>

## Our Approach
We will be focusing on Multimodal Learning with Neural Signals. Our proposal is that we can


### Related Works
In recent years, some efforts have been made to understand the connection between
the visual saliency content and the brain activity. In 2018, Zhen Liang et al. [Characterization of electroencephalography signals for estimating saliency features in videos.] presented
a model to study this connection and extracted sets of efficient features of EEG signals to
map to the visual salient related features of the video stimuli. The model has used the work
of Tavakoli et al. in 2017 [Bottom-up fixation prediction using unsupervised hierarchical models.]. The reconstruction of the features of the salient visual points
based on the features of the EEG signal has been performed with good accuracy in [44],
and prediction of the temporal distribution of salient visual points has been done using
EEG signals recorded in a real environment. In another study [Object extraction in cluttered environments via a
P300-based IFCE], the identification of the
objects in images recorded by robots was the purpose of the study, and a method based on
P300 wave was applied to identify the objects. The significant challenge for extracting the
objects of interest in navigating the robots is how to use a machine to extract the objects of
interest for humans. The combination of a P300-based BCI and a Fuzzy color extractor has
been applied to identify the region of interest. Humbeeck et al. [Presaccadic EEG activity predicts visual
saliency in free-viewing contour integration] have presented a model
for calculating the importance of the salient points for the fixation positions. Brain function
related to the extracted model has been studied using the eye-tracker and recording the
EEG signal. An evaluation of the connection between the importance of salient points and
the amplitude of the EEG signal has been done via this modeling. A multimodal learning
of EEG and image modalities has been performed in [Decoding brain representations by multimodal learning
of neural activity and visual features] to achieve a Siamese network for
image saliency detection. The idea of the work in [Decoding brain representations by multimodal learning
of neural activity and visual features] is the training of a common space of
brain signal and image input stimuli by maximizing a compatibility function between these
two embeddings of each modality. The estimation of saliency is achieved by masking the
image with different scales of image patch and computing the corresponding variation in
compatibility. This process is performed at multiple image scales, and results in a saliency
map of the image.





There seems to be only 3 sources with similar approach to visual saliency detection. We will explain the method used in each of them:

#### [Visual saliency detection guided by neural signals:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

Proposes a novel approach to visual saliency detection guided by brain signals. This method utilizes a two-step process:
* **Learning a Joint Brain-Visual Embedding:** This step involves training two encoders (one for images and one for EEG brain signals) to map them into a shared embedding space. The encoders are trained to maximize the similarity between the embeddings of corresponding images and EEGs. This aims to capture the relationship between what a person is seeing and their brain activity.  
* **Saliency Detection using Compatibility Variations:** Once trained, these encoders analyze how the compatibility between the EEG and image embeddings changes when different regions of the image are suppressed. Regions whose removal causes significant variations in compatibility are considered salient, resulting in a visual saliency map.
* This method combines deep learning's representational power with biological inspiration, aiming to learn visual saliency directly from brain activity.

#### [Visual Saliency and Image Reconstruction from EEG Signals via an Effective Geometric Deep Network-Based Generative Adversarial Network:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

Recent studies have shown that brain activity is impressed by visual saliency,the important parts of an image stimuli. 

In this paper, a deep model is proposed to reconstruct theimage stimuli from electroencephalogram (EEG) recordings via visual saliency. 

To this end, the proposed geometric deep network-based generative adversarial network (GDN-GAN) is trained to map the EEG signals to the visual saliency maps corresponding to each image. 

The first part of the proposed GDN-GAN consists of Chebyshev graph convolutional layers. The input of the GDN part of the proposed network is the functional connectivity-based graph representation of the EEG channels. 

The output of the GDN is imposed to the GAN part of the proposed network to reconstruct the image saliency.

The saliency metrics validate the viability and efficiency of the proposed saliency reconstruction network. 

The weights of the trained network are used as initial weights to reconstruct the grayscale image stimuli. The proposed network realizes the image reconstruction from EEG signals.

-------------------------

##### Techniques:

1. Chebyshev Graph Convolution

   - For detailed Explanation visit [here](https://github.com/ab-mahdi/EEG-AI-Salience/edit/main/isual%20saliency%20detection%20via%20learning%20a%20a%20shared%20brain-visual%20representation.md)

2. Generative Adversarial Network

  - Generative deep modeling is considered as an unsupervised learning task that discovers and learns the contents in input data in such a way that the extracted model can be used to generate new examples that could have been extracted plausibly from the original dataset.
  - For detailed Explanation visit [here](https://github.com/ab-mahdi/EEG-AI-Salience/edit/main/isual%20saliency%20detection%20via%20learning%20a%20a%20shared%20brain-visual%20representation.md)

3. Saliency Metrics

  -  Ground truth is necessary for calculating these metrics. Another input would be the saliency map. Considering these two inputs and computing these metrics, the degree of the similarity between them would be available.
  -  Similarity (SIM)
  -  Structural similarity (SSIM)
  -  Pearson’s correlation coefficient (CC)
  -  normalized scanpath saliency (NSS)
  -  The shuffled area under the ROC curve (s-AUC)
  -  SSIM interprets the structural similarity index using the mean and standard deviation of pixels of a selected window with fixed size in reconstructed image and the ground truth data, and it would bring a reliable measure of similarity.
  -  The s-AUC uses true positives and false positives according to the pixels of the reconstructed image in the locations of fixations in ground truth data, and is a confident metric of similarity between the two images.


''''
What is this???
The graph representation of the EEG records facilitates to exploit the functional connectivity between different channels in each EEG recordings in the classification procedure of the geometric deep network part of the proposed network. The overall model realizes the visual saliency reconstruction through the EEG records.

''''

Presents a Geometric Deep Network-based Generative Adversarial Network (GDN-GAN) for visual saliency detection and image reconstruction from EEG signals. This approach also uses a two-part architecture:
* **Geometric Deep Network (GDN) for Feature Extraction:** This part takes a graph representation of EEG channels as input, where the graph represents functional connectivity between the channels. The GDN extracts discriminative features from this graph to categorize the EEG signals into different visual patterns.
* **Generative Adversarial Network (GAN) for Saliency and Image Reconstruction:** The features extracted by the GDN are fed into the GAN. This GAN consists of a generator and a discriminator. The generator aims to create a saliency map from the EEG features, while the discriminator tries to distinguish between real saliency maps (derived from eye-tracking data) and those generated by the generator. Through this adversarial process, the generator learns to produce accurate saliency maps from EEG signals.
* The trained GDN-GAN can then be fine-tuned to perform image reconstruction from the EEG signals, going beyond just saliency detection.

------------------------------------------------

##### Network Architecture
The proposed geometric deep network-based generative adversarial network (GDN-GAN) architecture contains two parts of sequential layers.
Each part consists of a number of layers to map the EEG signals to the image saliency and to reconstruct the image stimuli.
- The trained weight vectors of the network parameters are used as initial weight vectors to train the network to map the EEG signal to the image stimuli and realize the image reconstruction from the brain activity.
- After functional connectivity-based graph embedding of the recorded visually evoked EEG signals, it imposed to the GDN part of the proposed network.


![Screenshot from 2024-08-13 13-07-14](https://github.com/user-attachments/assets/040d341f-7d66-4d71-a2ce-b203902ea2fe)

1. The GDN part extracts discriminative features of the different categories that the input belongs to.
  - ![helllll](https://github.com/user-attachments/assets/19f286ab-08e7-406a-9f3c-2a263bb7e7dc)
  - the detailed structure of the first GDN part of the network, and as it can be seen, it includes four layers of graph convolution. The Laplacian of the input graph is necessary to estimate the graph convolution of the input in each layer. The estimation is performed via the Chebyshev polynomial expansion of the Laplacian graph. Then, a batch normalization filters the output of each layer. After the fourth graph convolution layer, the
extracted feature vector is passed through a dropout layer. Then, the flattened output of the dropout layer is fed to a dense fully connected layer, and a log-softmax function is used for the classification of the output of the fully connected layer.
  - Shows The weights are trained to classify 40 categories of image stimulation and the flattened vector before the last dense layer is used to impose to the next GAN part of the network. The dimension of the flattened vector is equal to 6400.
  - ![image](https://github.com/user-attachments/assets/4c010da5-314f-4229-9e1a-a14330eebcc2)
  - illustrates the differences in the dimensions of every layer of the GDN. As each of the recorded EEG signals includes 128 channels, the constructed graph as input to the proposed GDN part in Figure 2 has 128 nodes. Every node in the constructed graph includes 440 samples. The input dimension of the graph convolutional layer independent of the number of graph nodes is considered to be 440, equal to the number of samples in each node. The obtained graph with the first graph convolution has 128 nodes with 440 samples in each vertex. A graph with 128 nodes with 220 samples in each vertex is the output of the second graph convolution, and the output of the third graph convolution operation is a 128-node graph with 110 samples in each of the nodes, and accordingly, the graph output of the fourth layer has 50 samples in each node. The attained 128-node graph with 50 samples in each node outputs a vector with 6400 elements.
  - The flattened vector is passed through a dense layer and the dimensions of the inputs and outputs of the dense layer are 6400 and 40, respectively.
  - ![image](https://github.com/user-attachments/assets/a482185e-b43d-4de6-a5be-9b137bfb9349)
  - Table 1 shows the dimensions of weight tensors for different layers of GDN part of the proposed GDN-GAN. Moreover, it shows the total number of parameters of graph convolutional layers according to the order of the Chebyshev polynomial expansion considered for each layer.
  - ![image](https://github.com/user-attachments/assets/32bf9274-27fd-4b9a-97d6-9af465323709)
  - ![image](https://github.com/user-attachments/assets/5a5fc379-5c04-40d7-ad6c-37c10dcbbf46)
  - ![image](https://github.com/user-attachments/assets/952c63c0-a7c3-45d8-a946-e1dfbe2655e0)
  - Figure 5 illustrates different layers of the GAN part of the proposed network.
  - Tables 2 and 3 give information about the details of the generator and discriminator parts of the proposed network, respectively.
  - **Training Procedure**: In order to fit the proposed GDN part to the EEG-ImageNet dataset, a training procedure is implemented, and the parameter weights of the network are optimized.
    - A 10-fold cross-validation strategy is used to train and evaluate the proposed network.
    - A standard gradient descent (SGD) is used to optimize the proposed GDN in each iteration, and the optimum parameters of the GDN are determined with the convergence of the train and test accuracy.
    - The trained weights of the GDN are transfered to the GDN-GAN to train the reconstruction part of the network.

2. The GAN part maps the extracted feature vector to the image saliency.
  - ![image](https://github.com/user-attachments/assets/e2b346e7-ad34-4451-a9aa-7ed43a04121a)

  - The generator part of the GAN consists of two dense layers, followed by four sequential transposed two-dimensional (2D) convolution layers, and one 2D convolution layer and leaky rectified linear unit is used as the activation function of all layers except for the first dense layer. The output of the GDN is imposed
to the generator, the input dimension of the generator is equal to 6400, and the output dimension of the first layer is 100. The output dimension of the second dense layer is equal to 20,000. The reshape layer converts the shape of the 20,000-dimensional vector to a three- dimensional output to impose to a 2D convolutional layer. Eight two-dimensional output with (50, 50) dimensions are imposed to the first transposed two-dimensional convolution layer. The kernel size in each of the transposed convolutional layers is equal to 4 × 4, and
the number of filters in each of them is equal to eight. The size of the strides in the first transposed convolution layer is equal to 2 × 2, in the second transposed convolutional layer, it is equal to 3 × 3 , and in the next two transposed layers, it is equal to 1 × 1. The output of the fourth transposed convolution 2D is imposed to the 2D convolution layer. The kernel size of the 2D convolution layer is considered as being equal to 2 × 2 , and the ize of the strides in this layer is equal to 2 × 2 . The output dimension of this layer is
equal to (299, 299), and is imposed to the last reshape layer. The output of the generator is a 299 × 299 -dimensional image. The schematic view of the outputs of each layer and the differences in the dimensions of the generator part of the proposed GDN-GAN are illustrated in Figure 6.
  - The adversarial part of the proposed GAN has three 2D convolution layers with the rectified linear unit as the activation function. The size of the kernel for each of these convolutional layer is considered equal to 4 × 4 , the size of the strides is equal to 2 × 2, and the number of filters for each of them is equal to four. The output of the third 2D convolution is flattened and imposed to a dense layer with an output dimension that is equal to one, to discriminate between fake or real images generated by the generator part of the GAN. Figure 7 illustrates schematic view of dimensions of different layers and it presents a tangible view of the outputs in each phase of the network.
  - ![image](https://github.com/user-attachments/assets/cb31d582-c249-4c97-8299-2c973c1aa29f)
  - **Training Procedure**: Binary cross-entropy is used as a loss function for the GAN part.
  - Discriminator loss is considered as the sum of the loss of the original image and the loss of the generated image.
    - For the loss of the discriminator output of the original image, instead of the ones vector as reference for calculating the cross-entropy between the reference and the original image, 0.9 is used as the coefficient of the ones vector.
    - For the loss of discriminator output of the generated image, the cross-entropy is calculated between the generated image and the zeros vector with dimensions equal to the generated image.
  - Generator loss is considered as the cross-entropy between the generated image and the ones vector with dimensions equal to the generated image.
  - An Adam optimizer with a learning rate equal to 0.0001 is used to train both the generator and discriminator networks.
  - The tuning of different parameters of the proposed GDN-GAN is achieved through a trial–error procedure. Training is performed with the use of different parameters available in Table 4 as a search space. The optimal values for training with good convergence are represented in this table.
  - ![image](https://github.com/user-attachments/assets/e1eafd34-1748-4669-913c-fa4abf2e07f5)

3. Overview
  - Figure 8 presents an overview of the proposed method for image reconstruction using the trained network for realizing the saliency map reconstruction. As it can be seen in this figure, the weights of the network are initialized with the pre-trained weights of the saliency map reconstruction scenario. Fine-tuning the transfered weights realizes the image stimuli reconstruction
  - ![image](https://github.com/user-attachments/assets/a152fb8b-c5a0-487e-99aa-14defd141066)

4. Results:

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

  - Figure 11a shows the training/test accuracy of GDN part of the proposed GDN-GAN with respect to the number of iterations in this network for the classification of 40 different categories of visual stimuli.
  - ![image](https://github.com/user-attachments/assets/05dfd744-dda8-4d4f-88a9-979c4d04a3ba)
  - Figure 11b shows the training/test loss function variations with respect to the number of iterations in this network for the classification of 40 different categories of visual stimuli.
  - ![image](https://github.com/user-attachments/assets/914a517d-eb54-40bb-89ca-0e57769f0cd5)
  - Figure 12 shows the receiver operating characteristic (ROC) plot for the GDN part of the proposed GDN-GAN and other state-of-the-art methods for classification of the EEG-ImageNet dataset, including region-level stacked bi-directional LSTMs [56], stacked LSTMs [57], and Siamese network [48] . The superiority of the GDN in terms of the area under the ROC can be seen in this figure compared to the other existing methods.
  -![image](https://github.com/user-attachments/assets/38048a64-25a2-416f-b926-10adda2ecdf0)
  - Furthermore, the performance of the GDN against the above-mentioned state-of-the-art methods in terms of precision, F1-score, and recall metrics is shown in Table 5.
  - ![image](https://github.com/user-attachments/assets/abdb4f39-593e-4540-939c-3c942c644e15)
  - To demonstrate the efficiency of the GDN, we compare the performance of our method with traditional feature-based CNN and MLP. For this purpose, three hidden layers for MLP and CNN with a learning rate of 0.001 have been considered. Maximum, skewness, variance, minimum, mean, and kurtosis have been used as feature vectors for every single channel.
  - According to Figure 13 , feature-based traditional deep networks such as MLP, CNN, and feature-based GDN have a poor performance in the case of classification of the EEG-ImageNet dataset with 40 different categories.
  - ![image](https://github.com/user-attachments/assets/1209a529-da56-4441-9de0-8b6362a276b4)
    - This figure shows the obtained accuracy of feature-based MLP, CNN , and GDN in 50 iterations. This figure illustrates that feature based GDN, MLP, and CNN have shown relatively similar performances, and the training times per epoch for these methods are very high for these traditional methods.
    - This figure confirms the good efficiency of the proposed GDN against the traditional
feature-based deep networks. According to this figure, the proposed network has high
classification accuracy compared to the other networks.
    - A good confirmation to the performance of the proposed method is the confusion matrix shown in Figure 14 .
    - ![image](https://github.com/user-attachments/assets/fe707469-d12f-4b85-81e6-91a78cbf9bfe)
      - The confusion matrix is an appropriate illustration of the performance of a network on test splits in the case of multi-class classification.
      -  Figure 14 shows the confusion matrix of the GDN part of the proposed method. This figure confirms the good performance of the classification part of the GDN-GAN. For the reconstruction phase of the proposed GDN-GAN, the training and evaluation of the proposed GDN-GAN are conducted according to the 10-fold cross-validation.
    -  The ground truth data are obtained using open-Salicon. The open-Salicon has been implemented using compiled the Python-compatible Caffe environment. The saliency evaluation metrics of the proposed GDN-GAN for different categories of visual stimuli are reported in detail in Table 6.
    -  ![image](https://github.com/user-attachments/assets/f970b2de-45c7-4994-9a4e-37fe9d94bd11)
      -  This table illustrates the saliency evaluation metrics according to the proposed method.
      -  The EEG signals are categorized in first part of the GDN-GAN. According to the extracted label, image stimuli is determined, and this image with the extracted feature of the first phase of the proposed method is imposed to the GAN part of the network to map the EEG signal to the saliency map of the image stimuli.
      -  After training, to test the GAN part, the EEG signals are imposed to the GDN-GAN, and the extracted images are compared to the original ground truth data through different saliency evaluation metrics, and the average of these metrics are reported in this table according to each category.
      -  Furthermore, the overall SSIM, CC, NSS, and s-AUC are represented through computing of the average of the saliency evaluation metrics among all categories.
  - According to this table, the proposed category-level performance of the visual saliency reconstruction method is over 90% except for six categories including Revolver, Running shoe, Lantern, Cellular phone, Golf ball and Mountain tent, in terms of SSIM and s-AUC.
  - SSIM interprets the structural similarity index using the mean and standard deviation of pixels of a selected window with fixed size in reconstructed image and the ground truth data, and it would bring a reliable measure of similarity.
  - The s-AUC uses true positives and false positives according to the pixels of the reconstructed image in the locations of fixations in ground truth data, and is a confident metric of similarity between the two images.
  - Considering these details, SSIM and s-AUC illustrate the limitations of the proposed GDN- GAN. However, considering the detailed values of the four saliency metrics, this table shows that the proposed GDN-GAN is a reliable and efficient method to map the EEG signals to the saliency map of the visual stimuli.
  - The trained GDN-GAN for saliency reconstruction is fine-tuned for image construction
issues. The loss plots in the result of training the generator and discriminator networks
for visual saliency and image reconstruction are represented in Figure 15 for three number
of categories. In addition, the SSIM and CC plots of both visual saliency and image
reconstruction per epoch for these categories can be seen in this figure.
The loss plots corresponding to both the saliency reconstruction and image recon-
struction illustrate that the variations in the generator and discriminator loss plots tend to
oscillate around one, as saliency evaluation metrics, including SSIM and CC, start to con-
verge ˙These are the behaviors of GANs, and these plots are confirmation of the effectiveness
of the proposed reconstruction of the GDN-GAN.
The results of visual saliency and image reconstruction for all of the 40 categories of
image stimuli are illustrated in Figures 16–19. In addition, the ground truth data and the
gray-scale versions of the original input image stimuli are shown in these figures. The
visual evaluations of these figures besides the saliency evaluation metrics confirm the
efficiency of the proposed GDN-GAN.
A comparison of the proposed GDN-GAN with state-of-the-art methods for image
saliency extraction is conducted, and the performance metrics are reported in Table 7.
The results of SalNet [ 43 ], SALICON [42], visual classifier-driven detector, ref. [ 48 ] and
neural-driven detector [48] are demonstrated in this table.
SALICON and SalNet are valuable approaches, considering the image data for saliency
map extraction according to the eye-fixation points of the eye tracking process while a
subject looking at an image. Another valuable approach, the visual classifier-driven detector
and the visual neural-driven detector by Pallazo et al. [48] , merges two modalities of EEG
signals and image data to extract the image saliency map efficiently. Our proposed GDN-
GAN is the first method that maps the EEG signals to the corresponding saliency map of
the visual stimuli and reconstructs the saliency map and image stimuli. Considering the
metrics according to these state-of-the-art methods concerning saliency map extraction in
Table 7, this confirms the efficiency of the the proposed GDN-GAN.
  - ddd
  - ddd
  - dd

#### [Salient arithmetic data extraction from brain activity via an improved deep network:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

Proposes a Convolutional Neural Network-based Generative Adversarial Network (CNN-GAN) for identifying and extracting arithmetic content (digits 0-9) from visually evoked EEG signals. The method involves:
* **CNN for EEG Classification:**  A CNN is trained on preprocessed EEG signals to classify them into 10 categories, representing digits 0 to 9. 
* **CNN-GAN for Saliency and Image Reconstruction:** The output of the trained CNN is used as input to a GAN, similar to Source 2. This GAN is trained to reconstruct both the salient parts of the MNIST digit images (using SALICON-generated images as ground truth) and the original MNIST digit images themselves. 
* The approach aims to extract not only the category of the visual stimulus but also reconstruct the image itself, particularly the salient regions, directly from EEG data. 




## Other Resources

Powerpoint at google slides
continue reading Visual Saliency Detection guided by Neural Signals

Check thise papers:
[Deep Learning Human Mind for Automated Visual Classification](https://arxiv.org/pdf/1609.00344)
[Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features](https://arxiv.org/pdf/1810.10974)
[Research on Multimodal Visual Saliency Detection Based on BP Neural Network Algorithm](https://ieeexplore.ieee.org/document/10236637)
[Multimodal contrastive learning for brain–machine fusion: From brain-in-the-loop modeling to brain-out-of-the-loop application](https://www.sciencedirect.com/science/article/abs/pii/S1566253524002252?via%3Dihub)
[research rabbit](https://www.researchrabbit.ai/)
[Brain-Machine Coupled Learning Method for Facial Emotion Recognition](https://ieeexplore.ieee.org/document/10073607)
[Object classification from randomized EEG trials](https://ieeexplore.ieee.org/document/9578178)
Find EEG-Eyetracking Datasets alsocheck kaggle
[An EEG & eye-tracking dataset of ALS patients & healthy people during eye-tracking-based spelling system usage](https://www.nature.com/articles/s41597-024-03501-y)
