# COS 470 Assignment 2
## File description
* **UnigramModel.py** - contains code for the unigram model and the main method to test the unigram model
* **BigramModel.py** - contains code for the bigram model and the main method to test the bigram model
* **MixedModel.py** - contains code for the mixed model, scripts to find the best lambda, and the main method to test the mixed model
* **utils.py** - contains utility scripts used in the models and main method
* **main.py** - main script for evaluating all models

## Run Instructions

* Place the Lyrics folder and the test.tsv file in the same directory as the submitted files 
* To evaluate all models and obtain f1 scores and significant testing run the main.py
* To test each model individually and see its predictions run the main() function in desired model file 

## Notes

* All scripts will create a new file called test2.tsv which contains corrected test cases (explanation can be found in the report)
* All scripts will create a new file in each genre directory called combined_text.txt which contains all lyrics of all songs in select genre