[https://developer.apple.com/documentation/createml/creating-an-image-classifier-model](https://developer.apple.com/documentation/createml/creating-an-image-classifier-model)

# Resources to find your dataset

[https://www.kaggle.com/](https://www.kaggle.com/)

# Farm Animal Classifier Project

We’ll be looking into how to create our own very own CoreML, model that classifies farm animals using CreateML. A CoreML model is created when we train it on a dataset of images we would like to recognize. We’ll be training our CoreML model on a dataset of farm animals. The objective is to teach our model to come up with rules to identify farm animals similar to those we give it in out dataset. 

# Convolutional Neural Network

Our CoreML model uses a convolutional neural network (or CNN) to learn the rules it needs to classify things. A convolutional neural network is an algorithm that extracts useful features about images and uses those features to classify images. There are two parts of this:

1. **The Convolutional Base -** These are the useful features that the CNN extracts to classify images.
2. **The Classifier** - Our CNN feeds the features into a classifier which groups together the similar features into distinct classes. 

# Transfer Learning

The process of training a machine learning from scratch is surprisingly difficult, expensive and cumbersome. Your first need a very very large dataset of images. A CNN model called SqueezeNet for example was trained on more than a million images to classify things in 1000 categories.

Transfer learning is a shortcut to training a CNN model from scratch by using an existing pre-trained model. The idea is to reuse knowledge from an existing model and adapt it to work on a specific task.

An existing model will have useful features it knows about from the dataset it was trained on. Therefore, we use those features and ignore the classifier that the pre-trained model comes with because that was built for a different task. We still need a dataset but we only need less than a few hundred images in many cases to make a perfectly reliable CNN Model.

Transfer learning works by once again extracting useful features but a new classifier is created. It heavily relies on the features it already knows about from the pre-trained model therefore the pre-trained model should have been trained on images fairly similar to those in our dataset.

# Introducing CreateML

Our objective is to be able to classify between three different types of farm animal: sheep, chickens and cows. CreateML makes this very easy. CreateML is an Apple framework that uses transfer learning to create custom machine learning models using Swift and macOS playgrounds.

### Credit

We start with a dataset of sheep, chickens and cows. CreateML uses transfer learning so we do not need an extremely large dataset. We need to ensure all images are in their own folder and correctly labelled. For example, we put all our chicken images in a folder named “chicken”. The actual image name does not matter. We will be using over 1000 images per folder. As a rule of thumb, 1000 images per folder is considered to be a large dataset. In general, the larger your dataset, the better your machine learning results will be. We do not need the exact same number  of images per folder but we need to ensure the disparity is not too large or else our model won’t be reliable.

**CreateML** makes training our models as simple as drag and drop. When the actual training begins, CreateML uses transfer learning to create our model. This means CreateML is using some kind of pre-trained model in the background. Its name is **VisionFeatureScreen_Print.VisionFeatureScreen_Print** was trained on a large dataset to recognize many things like animals, people, objects, and so forth. Since we are trying to classify farm animals our dataset is not too far off from what VisionFeatureScreen_Orint was trained on which is important if we want to get reliable results. Unfortunately we do not get to choose what pre-trained model we can use with CreateML. Other models like SqueezeNet and RestNet50 were perfectly good candidates to use in our situation but CreateML limits use to using VisionFeatureScreen_Print.

**VisionFeatureScreen_Print** looks for 2048 features in our images and assigns weights or numbers to each feature. A classifier known as a logistic regression classifier is used to classify each image. The logistic regression classifier will basically create a boundary called a hyperplane that separates our different farm animals into distinct classes in a multi-dimensional space. When our model attempts to classify something it basically looks at the features of an image and plots it in the multi-dimensional space to predict what it is.

# Finding and Curating a Dataset

Special credit goes to Corrado Alessio whose dataset we will be using to train our CoreML model. You can find the full dataset on Kaggle. We will be limiting our training to only three animals in this dataset. I have split the images into training, test and validation folders.

### Where Do I Find Images?

When you create your own machine learning you’ll most likely to create your own dataset. There are many sources to find our own images. I found this resource very useful.

Having an image source is just half the battle. **Google Open** **Images** for example only contains a list of links to images. One popular way to extract images from links is to use a Python script. This method is beyond this course. Please refer to the further reading links for more in depth material on creating a Python script to extract images. 

### Curating Images

The higher the quality of our images that the better the machine learning result we get. There are some common sense steps we should remember when curating our images. For example, 

- We should ensure our images are not too small.
- Images should also not have the same bias. For example, we should not have all the objects in our images appear at night.
- You will want the objects in your images to appear in different situations. Simply having the object you want to classify on its own is not a good idea.
