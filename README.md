# Object Reidentification
 
---

## Problem Statement

**Challenge Track 2 :  City-Scale Multi-Camera Vehicle Re-Identification**

Perform vehicle re-identification based on vehicle crops from multiple cameras placed at multiple intersections. This helps traffic engineers understand journey times along entire corridors.

---

## Dataset

Dataset : [Link](http://www.aicitychallenge.org/track2-download/)

The dataset contains 56,277 images, where 36,935 of them come from 333 object identities form the training set and 18,290 from the other 333 identities in the test set. An additional 1,052 images are used as queries. On average, each vehicle has 84.50 image signatures from 4.55 camera views.

Content in the directory:

1. "image_query/".  This dir contains 1052 images as queries. 
2. "image_test/".   This dir contains 18290 images for testing. 
3. "image_train/". This dir contains 36935 images for training. 
4. "name_query.txt". It lists all query file names.
5. "name_test.txt". It lists all test file names.
6. "name_train.txt". It lists all train file names.
7. "test_track.txt" & "test_track_id.txt". They record all testing tracks. Each track contrains multiple images of the same vehicle captured by one camera.
8. "train_track.txt" & "train_track_id.txt". They record all training tracks. Each track contrains multiple images of the same vehicle captured by one camera.
9. "train_label.xml". It lists the labels of vehicle ID and camera ID for training.
10. "train_label.csv". It lists the labels of vehicle ID in CSV format for training. 
11. "tool/visualize.py" & "tool/dist_example/". It is a Python tool for visualizing the results of vehicle re-identificaition, with an example of input data provided. 
12. "DataLicenseAgreement_AICityChallenge.pdf". The license agreement for the usage of this dataset.

---

## Evaluation Task


Find the image(s) in the test set that are from the same identity as the objects in each query image particularly, a list of the top 100 matches from the test set for each query image.


---



## Approach

We decided to solve this challenge using two approaches detailed below.


1) **Image Classifiers**

We are given 36935 training images where each image corresponds to one of the 333 classes. We will train an image classifier with ResNet50 as architecture using different pretrained models using fast.ai. Now we have two ways of training image classifier.

- **ImageNet pretrained model** : This is a straight forward approach of using ResNet50 model trained on ImageNet dataset as pretrained model for image classifier. This classifier will classify training images into 333 classes.(*Easy, no?*)
    
- **Stanford Cars pretrained model** : Here we will train a seperate ResNet50 model on [stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) which contains 16185 car images to be classified into 196 classes of cars. We will use the pretrained model trained on stanford cars dataset instead of ImageNet(as done in the above method) to classify the training images from Nvidia AI City Challenge into 333 classes.
  
 
After we train classifier we use the penultimate layer (Linear layer of size 512) to extract features of size 512 for each test and query image. As evaluation task demands us to find top 100 test images matching to each query image in ascending order of distance, we will find the distances using three methods:

- **KNeighbours** :  We use the [KNeighbours](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) from sklearn library to fit all 512 features of 18290 test image for different values of k and use [kneighbours](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors) function on each query image to obtain 100 neighbours (as we need top 100 test image).

-  **Annoy** : We use [Annoy](https://pypi.org/project/annoy/) library to find 100 nearest neighbours for each query image.

- **Euclidean Distance** :  This is the simplest approach to obtain the distance for each query image. We find the distance for each query image with all test image using `numpy.linalg.norm` to get top 100 test images in ascending order of distances.


2) **Siamese Networks**

We will use siamese network with triplet loss as baseline. 

A siamese neural network consists of same network typically a CNN, which accept distinct inputs and generates embeddings as output. We add different loss functions to encourage brining similar images together and different images apart in embedding space. Siamese NNs are popular among tasks that involve finding similarity or a relationship between two comparable things. Siamese networks typically share the same base network (CNN) which can be used as embedding extractor i.e. 3 inputs for triplet loss which will generate 3 outputs of embedding size of 128 or 2 inputs for contrastive loss will generate 2 outputs of embedding size of 128. 


---

## Types of Siamese Networks

Siamese Networks come in two flavours: using triplet loss and another using contrastive loss. Let's visit each of them.


- **Triplets Loss**

Triplets, are basically 3 images (yeah, no brainer!) which are anchor, positive and negative. What are these? **The anchor is any example from dataset. The positive is any example other than anchor belonging to same class as anchor.** And finally, **negative is any example belonging to class other than that of the anchor. In triplet loss, what we do is sample lot's of triplets.** There are different ways to sample triplets (or known as mining triplets) and this is what makes and breaks triplet loss approach. For some distance on the embedding space d, the loss of a triplet (a,p,n) is

$$\mathcal{L} = max(d(a, p) - d(a, n) + margin, 0)$$


There are different kinds of triplets such as easy triplets, semi-hard triplets, hard triplets. **Easy triplets are triplets which have loss 0**, because $$d(a, p) + margin < d(a,n)$$ **Semi-hard triplets are triplets where the negative is not closer to the anchor than the positive, but which still have positive loss,** i.e. $$d(a, p) < d(a, n) < d(a, p) + margin$$ **Hard Triplets are triplets where the negative is closer to the anchor than the positive**, i.e. $$d(a,n) < d(a,p)$$


In the original Facenet paper, they pick a random semi-hard negative for every pair of anchor and positive, and train on these triplets and also according to paper, selecting the hardest negatives can in practice lead to bad local minima early on in training. "Additionally, the selected triplets can be considered moderate triplets, since they are the hardest within a small subset of the data, which is exactly what is best for learning with the triplet loss", according to [this](https://arxiv.org/abs/1703.07737) paper. So, there is no thumb rule but starting with semi-hard yields good results and poor results with easy triplets.


Now having decided what strategy to use for batching the triplets, comes the challenge of training on batch of triplets. There are two ways in which batches of triplets can be trained : (i) **Offline triplet mining, where we produce triplets offline, at the beginning of each epoch for instance.** We compute all the embeddings on the training set, and then only select hard or semi-hard triplets. We can then train one epoch on these triplets. (ii) **Online triplet mining, where we compute useful triplets on the fly, for each batch of inputs.** This technique gives you more triplets for a single batch of inputs, and doesn’t require any offline mining. It is therefore much more efficient. Details of each method are described in [this](https://arxiv.org/abs/1703.07737) paper.

 The motivation is that the triplet loss encourages all examples belonging to same class i.e. of one  identity to be projected onto a single point in the embedding  space.


- **Contrastive Loss**

This the simplest among the two. Here, we sample two pairs of images, one **positive** and another **negative** or also known as, **similar** and **different**  pairs unlike above where we sample triplets. The two pairs i.e. **similar which contains any two images belonging to similar class** and **different where any two images belong to different class.** So, we create a lot many such similar and different pairs and pass it to any CNN architecture without the head(classification layer) and use the dimension of penultimate layer or add any number of linear layers  to obtain embeddings of size, say 128. Different name for loss suggests, we are not using our typical classification loss such as NLL or cross entropy for classification as there is no classification layer. So, how can we train our model (as in, what to backpropogate)? The answer is contrastive loss. **Contrastive Loss is a distance-based loss where such losses try to ensure that semantically similar examples are embedded close together.** This is what makes them special for our case, where we want similar images to be present closer in the embedding space and push different pairs away. We can also train by adding cross entropy as loss function with labels as 0 for similar images and 1 for different images. To learn more about this loss function, please read the paper linked below.


---


## Experiments

As stanford cars dataset contains cars, we decided to use the model pretrained using stanford cars dataset, as our model for experiments.

### KNeighbours

We experiment with different values of k for fitting 18290 test features each of size 512.

- k = 333

- k = 100

- k = 10

We observe that k=333, k=100 and k=10 yield same results.

### Annoy

We experiment with different values of k for fitting 18290 test features each of size 512.

- k = 300

- k = 100

- k = 10

We observe that k=300 and k=100 yield same results and k=10 gives bad results than both.


### Eucledian

This gives same result as k=333.




---



## References

[CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification](https://arxiv.org/pdf/1903.09254)

[Person Re-Identification in Identity Regression Space](https://arxiv.org/pdf/1806.09695.pdf)

[Attention Driven Person Re-identification](https://arxiv.org/pdf/1810.05866.pdf)

[Signature Verification using a "Siamese" Time Delay Neural Network ](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf)

[Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

[Facenet paper introducing Triplets](https://arxiv.org/abs/1503.03832)

[Contrastive Loss paper](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

[In defense of Triplet Loss for Person Re-identification](https://arxiv.org/abs/1703.07737)

[Andrew Ng's Triplet Loss Lecture](https://www.coursera.org/learn/convolutional-neural-networks/lecture/HuUtN/triplet-loss)

[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)



---

## Credits

Dataset for AI City Challenge : Nvidia

Dataset for Stanford Cars : Stanford

Fastai Library

---


**Note** : For pretrained model and training on stanford cars dataset, refer `Stanford_Cars_fastai.ipynb` notebook in same directory.

---


