# nlp-uncertainty-ssl

## Previous work
Current SOTA for NLP and SSL is [Cross-View Training](https://www.aclweb.org/anthology/D18-1217) what we could do is re-implement this and only use the samples that the main model is very confident to train the auxilary predictors.

## 11 class multi label emotion prediction
This task involves predicting if a Tweet contain zero or more emotions from a set of 11 different emotions. This is the E-c (Ag2) dataset from [Mohammad et al. 2018](http://saifmohammad.com/WebDocs/semeval2018-task1.pdf), it came from the [SemEval 2018 task 1](https://competitions.codalab.org/competitions/17751#learn_the_details-overview). The English datasets consists of 6838, 886, 3259 tweets for the training, development, and test datasets respectively which makes a total of 10983 annotated tweets.

### Re-formatting all of the data into JSON format

We makes the assumption that you store this English dataset in the following relative directory `../english_emotion_dataset` where training, development, and test datasets can be accessed from the following paths respectively:
1. `../english_emotion_dataset/train.txt`
2. `../english_emotion_dataset/development.txt`
3. `../english_emotion_dataset/test.txt`

We then process the following files by tokenising them before processing them any further e.g. applying machine learning or further subsetting them. To perform this processing use the following script, this script will also transformer the dataset so that it is a JSON file where each instance can be read as a Python dictionary. Lastly the last file path argument to this script will write the overall dataset statistics which are the percentage of labels per sample to this file as a Latex table:

``` bash
python tokenize_to_json.py ../english_emotion_dataset/train.txt ../english_emotion_dataset/development.txt ../english_emotion_dataset/test.txt ./results/tables/original_english_emotion_dataset_stats.tex
```
Example of this JSON format is below where each JSON sample is on a new line:
``` json
{"ID": "2017-En-21441", "text": "\u201cWorry is a down payment on a problem you may never have'. \u00a0Joyce Meyer.  #motivation #leadership #worry", "tokens": ["\u201c", "Worry", "is", "a", "down", "payment", "on", "a", "problem", "you", "may", "never", "have", "'", ".", "Joyce", "Meyer", ".", "#", "motivation", "#", "leadership", "#", "worry"], "labels": ["anticipation", "optimism", "trust"]}
{"ID": "2017-En-31535", "text": "Whatever you decide to do make sure it makes you #happy.", "tokens": ["Whatever", "you", "decide", "to", "do", "make", "sure", "it", "makes", "you", "#", "happy", "."], "labels": ["joy", "love", "optimism"]}
```

### Checking the JSON data has the same statistics breakdown
To check that when converting the data into json that none of the data has been lost or destroyed we can run the following script to get the label distribution:
``` bash
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/train.json ../english_emotion_dataset/development.json ../english_emotion_dataset/test.json
```
Output:
``` python
{'anger': '36.1', 'anticipation': '13.9', 'disgust': '36.6', 'fear': '16.8', 'joy': '39.3', 'love': '12.3', 'neutral': '2.7', 'optimism': '31.3', 'pessimism': '11.6', 'sadness': '29.4', 'surprise': '5.2', 'trust': '5.0'}
```
Which should be the same as `./results/tables/original_english_emotion_dataset_stats.tex` and [table 2 in the original paper](http://saifmohammad.com/WebDocs/semeval2018-task1.pdf).

As a side note this script can work with any number of json files e.g. if you only want to know the label distribution for the training dataset this can be done like so:
``` bash
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/train.json
```
Output:
``` python
{'anger': '37.2', 'anticipation': '14.3', 'disgust': '38.1', 'fear': '18.2', 'joy': '36.2', 'love': '10.2', 'neutral': '3.0', 'optimism': '29.0', 'pessimism': '11.6', 'sadness': '29.4', 'surprise': '5.3', 'trust': '5.2'}
```


## Format the OntoNotes 5.0 data
The data can be found from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19). The instructions on how to format it can be found [here](https://cemantix.org/data/ontonotes.html) however these instructions are only valid up to step 3 due to the script not being avaliable from step. Therefore from step 3 onwards the scripts to download can be found [here](http://conll.cemantix.org/2012/data.html). From this you need to do the following:
1. Within `conll-formatted-ontonotes-5.0-12` folder create the following two directories `conll-2012` and `v4` and then put the `v4` folder within `conll-2012`.
2. Put the `data` folder that is within `conll-formatted-ontonotes-5.0-12` in the `v4` subdirectory.
3. Put the `scripts` folder that you downloaded from [here](http://conll.cemantix.org/2012/data.html) into the `v4` folder also.
4. At this stage the `conll-formatted-ontonotes-5.0-12` should look like this `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/data` and `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/scripts`
5. Assuming that the `ontonotes-release-5.0` folder downloaded from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) is in the same directory as `conll-formatted-ontonotes-5.0-12` run the following command from the `scripts` folder: `./skeleton2conll.sh -D ../../../../ontonotes-release-5.0/data/files/data/ ../../../conll-2012/`
6. That last command should have added the annotations from the ontonotes 5.0 data for English using the corrected text files from [here](https://github.com/ontonotes/conll-formatted-ontonotes-5.0/releases/tag/v12).

