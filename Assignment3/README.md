## Set up
* Run the following command to download the en_core_web_sm model (English language multi-task CNN) 
```
python -m spacy download en_core_web_sm
```
* Because of the extraction of some specific features, we needed to keep the training songs from Task 1 in separate files instead of a csv. To obtain the training files please go to https://github.com/jakubSuranUSM/COS-470/tree/master/Assignment3 and download the "Training Songs" folder and place it in the root project directory. 
* The test songs are also read from the files as it can be observed in the GitHub repository from above.

## Run
* To run this project, run the main method in the main.py file. The main.py file also contains a _**TRAIN**_ parameter which controls if the model will be trained from scratch or if the model's weights are going to be loaded from a file.
* **Note:** I accidentally replaced the saved model that I evaluated and wrote the report about. The f1 scores from the model and the report might differ.


## File description
* feature_extraction.py - contains scripts for feature extraction
* feature_utils.py - contains extraction scrips for all features 
* GenreClassifier.py - contains the model architecture and training/evaluation scripts
* SongFeaturesDataset.py - contains a dataset class that will be fed to the model
* main.py - main file, whole Task 2 pipeline

### List of Features for Song Classification from Lyrics:

**1. Word-Level Features:**

* Number of Words
* Vocabulary Size
* Average Line Length
* Average Word Length
* Part-of-Speech Distribution (Nouns, Verbs, Adjectives, etc.)
* Stop Word Ratio
* Named Entity Recognition (People, Places, Organizations) - decided to not use in the final model, did not display strong correlation to genre

**2. Rhyme and Structure Features:**

* Rhyme Density
* Line Length Variation

**3. Sentiment Analysis:**

* Overall Sentiment Score (Positive, Negative, Neutral)