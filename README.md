# Music-Genre-Classification-GTZAN
The project uses Machine Learning and Deep Learning techniques to Classify music into 10 genres of music as provided in the GTZAN dataset.

### Approach

#### 1) Machine Learning :-
Features like mel-ceptral coefficients, zero-crossing rate, power, loudness etc. are extracted from audio files using librosa library afterwhich the data is feed into the ML models to classify data.

#### 2) Deep Learning :-
Mel Spectrogram images are formed from audio files which are very strong features to discriminate the properties of two audio files. So, these images are feed into CNN model for classification into 10 genres.

### Read and follow the instructions carefully.

## Packages
* Tensorflow 
* Keras
* Scikit-Learn
* Tkinter
* Librosa
* Numpy, Pandas, Matplotlib
* Python 3.6.0


## Raw data:
Download GTZAN Dataset from: http://opihi.cs.uvic.ca/sound/genres.tar.gz
Raw data is 1.2GB and consists of 1000 audio files(.au) divided into 10 folders for 10 genres equally. I.e Every genre has 100 audio files.

## Preprocessed data
The raw audio has been converted to mel-spectograms and other features using librosa library for Machine Learning purpose and can be found in the folder as csv file. Feature spectrogram images are made from the audio files directly during training for Deep Learning model.


## Code Notebooks

### Machine Learning
To run code in any of these notebooks, first please download raw data from Dataset link above 
1. make_dataset_ml: Extracts features from dataset for ML training. 
2. audio_data_visualization: Plots time series visualization and spectograms for the 10 different genres and exports them as image.
3. train_classical_ml_models: Loads the generated feature file. Trains the dataset using lgbm+SVM and XGBoost architecture and exports the model after training.


### Deep Learning

1. CNN_train: This notebook contains all the steps right from creating images from audio to training them using CNN and finally exporting the model too.


## Working of Project

* First Make sure u have the CNN model in your models directory. If not, you can download it from : https://drive.google.com/file/d/1-HeGSucv1nlaxWhOsEFzPwLAO5UFfYyd/view?usp=sharing

* Run AudioClassification python script, play some music and let the model classify music genre for u ;)


## Videos of the Project

* Video for the Presentation of Making of Project can be found here : https://www.youtube.com/watch?v=QWGrpwO4O_k

* Video for Testing and Working of the Project can be found here : https://www.youtube.com/watch?v=f9_q2LIAbrM


## Conclusion

Deep Learning methodology proves to be more accurate with around 75% accuracy in classifying the genres whereas machine learning acheives around 65% accuracy. You can see the csv file in models directory where the comparison is made between used architectures.
