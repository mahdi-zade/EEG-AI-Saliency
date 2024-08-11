# EEG-AI-Saliency

## Abstract



## Introduction
Understanding how the brain processes external stimuli is crucial for advancing many fields. namely assistive tecnologies, cognitive neuroscience, education, sports, Consumer and Lifestyle Technologies, Marketing and User Experience. 
Electroencephalography (EEG), a widely used non-invasive brain recording technique, has been instrumental in studies exploring brain activity across various contexts, including attention, memory, motor control, and vision. 
Recent research has focused on mapping the relationship between EEG signals and visual saliency—the identification of key elements in a scene that stand out due to their visual distinctiveness or task relevance. Despite significant progress, the connection between EEG recordings and image saliency, particularly through the use of dynamic information from functional connectivity between brain regions, remains underexplored.

Advancements in deep learning have revolutionized computer vision, enabling remarkable improvements in tasks such as image classification and object detection. However, automated vision systems still lag behind human capabilities in interpreting complex visual environments. Saliency detection, a fundamental aspect of human visual perception, allows us to efficiently process scenes by focusing on the most important elements, thereby optimizing cognitive resources. Integrating deep learning techniques with EEG signal analysis has shown promise in enhancing our understanding of how visual stimuli influence brain activity. For instance, deep networks have been employed to classify EEG responses to visual inputs and even reconstruct images from brain signals, highlighting the potential of this interdisciplinary approach to bridge the gap between neural activity and visual perception.
There are many ways to classify various methods used for visual saliency detection.
The most recent review paper in the field has classified the mothods into two broad categories:
- Studies have shown a strong correlation between brain representations in the visual pathway and activation patterns within deep neural networks.
- This has led to research into "brain-guided" saliency detection, using neural signals like EEGs as an additional input or supervisory signal for deep learning models.

![image](https://github.com/user-attachments/assets/9a204af3-b439-452b-87ac-53710bbe8899)


## Conventional Methods for Saliency Detection
Traditional approaches to saliency detection aim to identify salient areas in images by analyzing hand-crafted features and designing algorithms that simulate human perception, often inspired by biological processes in the human visual system. These methods often analyzed low-level visual cues like color, intensity, and orientation to create saliency maps, highlighting regions that stand out from their surroundings

### Limitations 
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



## Deep Learning Methods for Saliency Detection

### Advantages

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


### Current Deep learning Approaches to saliency detection


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


#### Related Works
There seems to be only 3 sources with similar approach to visual saliency detection. We will explain the method used in each of them:

#### [Visual saliency detection guided by neural signals:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

Proposes a novel approach to visual saliency detection guided by brain signals. This method utilizes a two-step process:
* **Learning a Joint Brain-Visual Embedding:** This step involves training two encoders (one for images and one for EEG brain signals) to map them into a shared embedding space. The encoders are trained to maximize the similarity between the embeddings of corresponding images and EEGs. This aims to capture the relationship between what a person is seeing and their brain activity.  
* **Saliency Detection using Compatibility Variations:** Once trained, these encoders analyze how the compatibility between the EEG and image embeddings changes when different regions of the image are suppressed. Regions whose removal causes significant variations in compatibility are considered salient, resulting in a visual saliency map.
* This method combines deep learning's representational power with biological inspiration, aiming to learn visual saliency directly from brain activity.

#### [Visual Saliency and Image Reconstruction from EEG Signals via an Effective Geometric Deep Network-Based Generative Adversarial Network:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

Presents a Geometric Deep Network-based Generative Adversarial Network (GDN-GAN) for visual saliency detection and image reconstruction from EEG signals. This approach also uses a two-part architecture:
* **Geometric Deep Network (GDN) for Feature Extraction:** This part takes a graph representation of EEG channels as input, where the graph represents functional connectivity between the channels. The GDN extracts discriminative features from this graph to categorize the EEG signals into different visual patterns.
* **Generative Adversarial Network (GAN) for Saliency and Image Reconstruction:** The features extracted by the GDN are fed into the GAN. This GAN consists of a generator and a discriminator. The generator aims to create a saliency map from the EEG features, while the discriminator tries to distinguish between real saliency maps (derived from eye-tracking data) and those generated by the generator. Through this adversarial process, the generator learns to produce accurate saliency maps from EEG signals.
* The trained GDN-GAN can then be fine-tuned to perform image reconstruction from the EEG signals, going beyond just saliency detection.

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
