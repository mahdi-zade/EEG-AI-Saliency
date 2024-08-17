# Dataset:
- Contains 11,965 EEG sequences recorded while a group of 6 participant subjects looked at images displayed on a computer screen.
- The images are taken from a subset of 40 classes from ImageNet with 50 images per class (in total, 2,000 images).
- Each EEG sample has 128 channels, each one with about 500 values for each observed image.
- All EEG signals were first detrended in order to remove unwanted linear trends, filtered with a notch filter (50 Hz and its harmonics) and with a bandpass filter with low and high-cut frequencies, respectively, at 5 Hz and 95 Hz, and finally z-scored (zero-centered values with unitary standard
- Training, validation and test splits of the EEG dataset consists respectively of 1600 (80%), 200 (10%), 200 (10%) images with associated EEG signals, ensuring that all signals related to a given image belong to the same split.
- Eye fixations recorded using a 60-Hz Tobii T60 eye-tracker – of the same six human subjects who underwent EEG recording. Training, validation and test splits are the same of EEG data.
- [IMAGENET Database](https://mindbigdata.com/opendb/imagenet.html)
# Evaluation Metrics
- shuffled area under curve (s-AUC) 
- normalized scanpath saliency (NSS)
- correlation coefficient (CC)


# Encoding
- ![image](https://github.com/user-attachments/assets/7a09e3a8-1d8e-41da-a63a-d37f714606f6)

- EEG encoder: EEG-Net that is a state-of-the-art network for classifying EEG signals and consists of a sequence of a convolutional layers.
- Image encoder: Inception-v3 model, pre-trained on ImageNet, to extract visual features, followed by a linear layer for mapping to the joint embedding space.

# Training the encoders
- Comptibility function:
  - ![image](https://github.com/user-attachments/assets/e032ad29-81c1-4718-8149-b4191521cbef)
- Classification functions:
  - ![image](https://github.com/user-attachments/assets/b9c7dc1f-fa7d-4877-ab7d-b54cf3139a05)
  - In the equations above, V(y) is the subset of images in V with class label y; similarly for E(y).
  - Classification is performed by selecting, for an input item of one modality, the class whose corresponding items of the other modality yield the largest average compatibility. Clearly, then, it is necessary that F accurately matches EEG signals and images, which in turn depends on how close the embeddings provided by the respective encoders are.
- Structured Hinge Loss for Training
  - The encoders are trained by minimizing a structured hinge loss, which encourages the compatibility function to assign higher scores to correct pairs (same class) than to incorrect pairs (different classes).
  - ![image](https://github.com/user-attachments/assets/2747885a-6fbd-439e-91ef-76352b96ff89)
  - Minimize the function below:
  - ![image](https://github.com/user-attachments/assets/fbec8a58-7610-422c-8ec7-5f1412411016)

# Saliency Detection
- The saliency detection method identifies important features in the image by analyzing how the compatibility score changes when parts of the image are masked.
- Measure how the compatibility function F varies as image patches are suppressed at multiple scales.
- The most important features in an image are those that, when inhibited in an image, lead to the largest drop in compatibility score with respect to the corresponding neural activity signal.
- ![image](https://github.com/user-attachments/assets/be4166ad-335a-45d2-bd80-40c0a682b472)
- The contribution of pixel (x,y) to saliency at scale s, indicated as C(x, y, s, e, v), is computed by masking out a s×s patch around (x, y) pixel in the image v and, assessing the compatibility variation before and after removing it.
- s scales: 3, 5, 9, 17, 33, 65.
- Since we have the saliency contribution of each pixel, we can compute the output saliency map by summing up all pixel contributions and performing image-by-image basis normalization.

# Results
- In order to demonstrate that the obtained saliency is not only influenced by the image encoder and that neural signals, indeed, contribute to its prediction, we employed the baseline.
- We used Inception-v3 (used for joint learning) and applied the approach in Saliency detection part with the difference that the saliency score was not based on compatibility, but on the log-likelihood variation for the image’s correct class.
- ![image](https://github.com/user-attachments/assets/e00a3587-c53b-4cf8-ad21-f496a557ae5e)

Obtained performance

reveals that our method outperforms or is on par with state
of the art saliency detectors, without using any supervision
on the specific task, while those methods do. In addition,
the performance difference between our brain-driven saliency
detection method and the baseline indicates that the neural
modality has an active role in detecting visual saliency.
Fig. 3 shows qualitatively the saliency maps obtained by

our approach, compared to those achieved by state-of-the-
art saliency detectors [3], [4] and the baseline. The obtained

saliency maps demonstrate also how the employment of the
jointly-learned neural/visual features yields more accurate
maps than the methods under comparison, in terms of
localization quality and amount of noise in non-salient areas.
