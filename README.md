# COVID-CAPS
<h4> A Capsule Network-based framework for identification of  COVID-19 cases from chest X-ray Images </h4>

Novel Coronavirus disease 2019 pneumonia (COVID-19), with its relatively high intensive care unit (ICU) admission and mortality
rate, is rapidly spreading all over the world. Early diagnosis of this
disease is of paramount importance as it enables physicians to isolate
the patents and prevent the further transitions. The current gold standard
in COVID-19 diagnosis requires specific equipment, is timeconsuming,
and has relatively low sensitivity. Computed tomography
(CT) scans and X-ray images, on the other hand, reveal specific
manifestations associated with this disease. However, the overlap
between different bacterial and viral cases, makes the humancentered
diagnosis difficult and challenging.

The capsule network-based model (COVID-CAPS), proposed in this
study, for diagnosis of COVID-19, is capable of handling small
datasets, which is of significant importance as this disease has been
only recently identified and large datasets are not available.

So far, our results has shown that COVID-CAPS has competitive advantageous over
previous CNN-based models, on a public dataset of chest X-ray images. <br> 
COVID-CAPS structure and methodology is explained in detail at <a href="https://arxiv.org/abs/2004.02696">https://arxiv.org/abs/2004.02696</a> 

<h3>Note : Please don’t use COVID-CAPS as the self-diagnostic model without performing a clinical study.</h3>
  COVID-CAPS is currently a research model aiming to accelerate building a comprehensive and safe diagnostic system to identify COVID-19 cases. This model is not a replacement for clinical diagnostic tests and should not be used as a self-diagnosis tool to look for COVID-19 features without a clinical study at this stage. Our team is working on enhancing the performance and generalizing the model upon receiving more data from medical collaborators and scientific community. You can track new results and versions as they will be updated on this page.<br>

![Roadmap](CapsNet/model.png)

## Dataset
We used the same dataset as the one used <a href="https://github.com/lindawangg/COVID-Net">here.</a>
This dataset is generated from the following two publicly available chest
X-ray datasets.
* <a href="https://github.com/ieee8023/covid-chestxray-dataset">covid-chestxray-dataset</a> : A new data collection project to make COVID-19 data publicly available to the scientific community.
* <a href="https://www.kaggle.com/c/rsna-pneumonia-detection-challenge">RSNA Pneumonia Detection Challenge</a> : The dataset used in the Pneumonia Detection Challenge organized by the Radiological Society of North America.

As the main goal of this study is to identify
positive COVID-19 cases, we binarized the labels as either positive
or negative. In other words the three labels of normal, bacterial, and
non-COVID viral together form the negative class.

## Pre-Training
Our pre-training dataset consists of 94323 frontal view chest X-ray images for common thorax diseases. This dataset is extracted from the NIH Chest X-ray dataset available online for public access <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345">here</a>. Chest X-ray14 dataset originally contains 112120 X-ray images for 14 thorax abnormalities. This dataset also contains normal cases without specific findings in their corresponding images.
To reduce the number of categories, we classified these 15 groups into 5 categories based on the underlying relations between the abnormalities in each disease. The first four groups are dedicated to No findings, Tumors, Pleural diseases, and Lung infections categories. The fifth group encompasses other images without specific relations with the first four groups.
We then removed 17797 cases with multiple labels (appeared in more than one category) to reduce the complexity and downscaled all images from (1024,1024) to (224,224).

### Steps to prepare the pre-train dataset
1. Follow the guideline in the database folder to download and unpack the original dataset
2. Run xray14_preprocess.py
This will create another folder in the current directory named database_preprocessed to store downscaled images all in one place. The process may take several minutes.
3. Run xray14_selection.py
It will import preprocessed images in the previous step as numpy arrays and stack them together to form two numpy arrays named X_images and Y_labels. These two numpy arrays will then be used to pretrain the dataset.
Note: It would be impossible to allocate such a large space for a numpy array for some processing systems(based on CPU or GPU capacity and configuration). You may need to split the process into several steps and save the data into seperate numpy arrays and then combine them.


## Requirements
* Tested with tensorflow-gpu 2 and keras-gpu 2.2.4
* Python 3.6
* OpenCV
* Pandas
* Itertools
* Glob
* OS
* Scikit-image
* Scikit-learn
* Matplotlib

## Code
The code for the Capsule Network implementation is adapted from <a href="https://keras.io/examples/cifar10_cnn_capsule/">here.</a>
Codes are available as the following list:
* binary.py : Main code
* test_binary.py : Test and Evaluation
* weights-improvement-binary-86.h5 : Best model's weights without pre-training
* weight-improvement-binary-after-44.h5 : COVID-CAPS weights after fine-tuning
* pre-train.h5 : weights for the pre-trained model
* pre-train.py : codes for pre-training
* binary-after.py : codes for fine-tuning
* xray14_preprocess.py : Extracting and rescaling Chest XRay14 dataset
* xray14_selection.py : Converting downscaled Xray14 images into numpy arrays and saving them
