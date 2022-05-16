# Compression Based Perceiver.
This repository contains the Final Source Code for our final year project CS 492: Group 12 - Compression Based Perceiver at Habib University.
## Team Members:
1. Shams Arfeen. (sa05169@st.habib.edu.pk)
2. Umme Salma. (us04315@st.habib.edu.pk)
3. Ali Haider. (ar05199@st.habib.edu.pk)
4. Muhammad Munawwar Anwar (ma04289@st.habib.edu.pk)
## Supervisors:
1. Dr. Syed Sohaib Ali. (Supervisor, Assistant Professor, Computer Science - Habib University)
2. Dr. Muhammad Farhan (Co-supervisor, Assistant Professor, Electrical & Computer Engineering - Habib University ) 
3. Syed Nouman Hasany.  (External Supervisor, Erasmus Mundus Joint Master Degree in Medical Imaging and Applications - Universitat de Girona)
4. Syed Talal Wasim. (External Supervisor, Erasmus Mundus Joint Master Degree in Image Processing and Computer Vision - Universidad Aut ́onoma de Madrid)

## Project Abstract:
Google‘s Deep mind aims to build more general and versatile architectures that can handle all types of data. Transformers have the biggest gain on large scale problems. 
The quadratic complexity of the self-attention mechanism in Transformers makes it difficult to train the model on large scale datasets. There is a need for a model that 
can be trained on large scale datasets using a reasonable amount of computational resources. As a result, Deep Mind proposed a transformer-based architecture, Perceiver 
which scales linearly with input size and works on the principle of crossattention mechanism. Our research question is to evaluate the performance of the Perceiver Architecture 
for classification when it is passed a latent representation of the image as an input. Our hypothesis is that if the representations capture enough semantic information, 
the perceiver should be able to classify the embeddings with accuracy comparable to the raw input data.We used both supervised and unsupervised representation learning methods, 
Autoencoders and Supervised Contrastive Learning. The datasets we used were CIFAR10, CIFAR100, and a subset of the ImageNet dataset containing 60,000 images and the same classes as 
the CIFAR10 dataset. The obtained results have been evaluated quantitatively and qualitatively, and they establish that embeddings obtained using Supervised Constrastive Learning achieve 
competitive performance compared to the baselines. On the other hand, the Autoencoder based embeddings fall short of achieving the same performance. This is because the Autoencoder focuses 
on spatial consistency rather than on semantic information as is the case with Supervised Contrastive Learning. In the future, this work can be extended using multidimensional integration 
or by creating integrations by distilling knowledge. This study is also for those who are working to generate compressed input representations using different representation learning models.

## Showcase Website:
The showcase website for this project can be accessed from the following URL:

https://fyp-showcase.herokuapp.com/home

## ImageNet-10:
The ImageNet-10 is a custom subset of the ImageNet Dataset that was extract by us .This dataset contains 60,000 RGB images with dimensions of 224 by 224 and 10 classes with 6000 samples per class. 
Furthemore, the train test split has been done to make sure that the dataset is balanced and can be found in their respective folders. The images belonging to each class are stored in their respective folders. 
Each filename has the following format: <class_name>_<image_number>.jpg. The ImageNet-10 can be downloaded from the url below:

https://drive.google.com/drive/folders/1sOZFQaDO7SlU_b_U-WCjcWns3DTGfrjl
