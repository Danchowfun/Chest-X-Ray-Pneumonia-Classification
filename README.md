# Chest-X-Ray-Pneumonia-Classification
I. Introduction
The year 2020 is very special to everyone in the world. As the explosion of COVID-19 pandemic, people’s life are largely affected. Due to the high infectious rate of coronavirus, the accurate and quick detection of COVID-19 patients become necessary. However, with the limited medical resource, this become increasingly difficult when there are a large amount of patients.
Therefore, we want to use Machine Learning algorithms to study the chest x-ray images of the patients. We want to create a model that can help detect the suspected COVID-19 or pneumonia patients and speed up the physicians’ diagnose process through highlighting the suspected area of the chest x-ray.

II. Dataset
We obtain two datasets Chest X-Ray Images (Pneumonia) and covid19-pneumonia-normal-chest-xray-pa-dataset from Kaggle as our data.

III. Experiment


(i) Object of study
We want to perform classification on patients' chest-xray images to help physicians separate Normal, Penumonia and COVID-19 patients to speed up the process of diagnosing COVID-19 patients and help control the coronavius pandemic.



(ii) Clustering
Before applying supervised machine learning algorithms, we created t-SNE and PCA visualizations to know more about our dataset, and t-SNE turns out to have a better performance in showing clusters. However, we discovered that part of Normal class locate between pneumonia and COVID in the visualization. It implys us that we need to use data augmentation to minimize the influence. We also noticed that there are slight differences between our test and train datasets



(iii) Algorithems implemented
In this research, we used various machine learning models and deep learning models to do the classification, including: (1) Histogram of Oriented Gradients (HOG) + KNN, SVM, Logistic Regression; (2) Histogram of Oriented Gradients (HOG) + SVM; (3) Histogram of Oriented Gradients (HOG) + Logistic Regression; (4) Convoluted Neural Network; (5) Resnet 18; (6) Resnet 50; (7) Generative Adversarial Networks 
1. Histogram of Gradients
Since our dataset include chest-xray images exclusively, using deep learning models might logically be the best choices. However, we are still curious about how traditional machine learning models would perform when classifying images. Therefore we chose to use Histogram of Oriented Gradients (HOG) to extract features from our images and input those features into KNN, SVM and Logistic regression to do the classification and compare the accuracy.
2. Convolutional Neural Network
At the beginning of our project, each of us has tried to build some CNN models with Tensorflow or Pytorch, and the accuracies range from 75% to 85%.  For the data processing, we used data augmentation because we only have about 6000 images. One of our models which has a good accuracy is a simple sequential model, starting with a convolutional network of kernel size (3,3) and max pooling with pool size (4,4) and a dropout rate of 0.2, followed by another convolutional network of kernel size(3,3) and max pooling size of (8,8) and a dropout rate of 0.2. In this model, we also set a early stopping with patience of 5 to prevent overfitting. We split 25% of training data into validation set and set batch size as 300 and run for 40 epochs. 
3. ResNet
We imported resnet 18 and resnet 50 from trochvision models. By applying a couple of transformations to our training data, like resize, center crop, rotation and horizontal flips we can easily make our dataset even larger and help us to create a robust model. Regularization is also used to help us avoid overfitting. We simply set the weight_decay = 0.1 in the optimizer. In our training, we used Crossentropy Loss and choose to use SGD as optimizer. It is a basic algorithm responsible for having neural networks converge. The motivation behind SGD is that it only calculates the cost of one example for each step instead of using all training examples in the dataset. In this case, it speeds up the whole training process greatly. We started the training with a relatively large learning rate and then decayed the learning rate by a factor of 0.1 in every 10 epochs. By doing this, the neural network learnt faster in the first half of our training process and by decreasing the learning rate later, it ensured that the network can learn a more optimal set of weights in the second half.
4. Transfer Learning
Since we have very limited data amount, other than training the renset using our own dataset, we also tried pretrained Resnet 50. Here we use the resnet 50 pretrained by Imagenet dataset. We extracted the weights or convolutional layers that obtained by training resnet 50 on Imagnet dataset and used the weights to extract features from our chest_xray image dataset. Then we feed those features into a fully connected layers and perform the classification. 
5. Generative Adversarial Networks
	Normal	Pneumonia	COVID-19
Real			
Generated			
Table 1. Comparison between real and generated images
After learning about Generative Adversarial Networks, we've been thinking if we can apply GAN on our dataset to increase the size of our data and improve the accuracy. After doing some researches, we found that the conditional Generative Adversarial Networks is an appropriate solution for our problems. Therefore, we learning from a Fashion MNIST cGAN example and modified it to apply it on our dataset. We've encountered some problems when tuning the parameters. For example, due to the OOM problem, we have to choose the node size as 25*25 in order to generate 100*100 images, and there will be clear signs of blocks on the images when the number of epochs is low. The signs became invisible after we increase the number of epochs to 80, but the training process took nearly an hour.
The generated images in Table 1 is obtained after 100 epochs of training. We can see the difference between different classes. For example, the lungs of normal people are very clear when the lungs of pneumonia patients have shadow on them. 
We generated 5000 images after training the models and combine the generated images with the real images. Then, we applied our CNN model on our new data, and the results are satisfactory because it helps us reach 90% accuracy.

IV. Results
Algorithms	Data Augumentation	Epoch	Accuracy
HOG + KNN, SVM, Logistic Regression	No	\	KNN: 73.0%, SVM: 79.9%, Logistic Regression:81.4%
CNN	No	40 epochs	83.3%
CNN	Yes	40 epochs	85.5%
Resnet 18	Yes	60 epochs	87.07%
Resnet 50	Yes	70 epochs	88%
Resnet 50 pretrained by Imagenet	Yes	10 epochs	89%
CNN on cGAN generated images	No	40 epochs	90.3%
Table 2. Classification accuracies from algorithms implemented.
From the accuracy table, we can see that the highest accuracy is 90.3% by combining original data with our GAN generated images and applying CNN on the data. When we exclude the influence of GAN generated images, ResNet has the best performance among all models that we have tried. In this case, the traditional machine learning models performed a bit inferior with accuracies around 70% to 80%.

V. Conclusion
(i) Project Summary
In this project, we've conducted both supervised and unsupervised machine learning algorithms on our dataset. We gradually improved our accuracy from 70% to 90% by testing on various machine learning models above.
Looking at accuracies different models attained, we see that deep learning models are performing better than machine learning models like KNN or SVM. We also reassure that Conditional Generative Adversial Network can help train our models and increase accuracy, even in the case of medical chest_xray. 
(ii) Application
In our dataset, we have 3875 pneumonia, 1341 normal, and 1500 COVID-19 chest X-ray images. The large proportion of pneumonia class in the dataset leads to a very high accuracy of pneumonia patient detection (99% in some models) and relatively low accuracy in the other 2 classes.
With a high accuracy of pneumonia detection, we would only conduct nucleic acid test on people classified as pneumonia patient, and those misclassified normal people would then be corrected, and it would save a huge amount of time, detection kits and efforts for medical department.
(iii) Further Improvements
More data will be very helpful, especially COVID-19 chest X-ray images, since we only 1500 COVID images when some of those are transformed from other images in the dataset. In addition, from the t-SNE visualization we can see that COVID-19 class is more seperated from the other two classes, so there might be some other features except the disease affecting our classification. Therefore, we need more data to improve our classification.

