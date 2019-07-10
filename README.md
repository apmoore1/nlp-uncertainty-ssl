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

### Running the emotion models
Below we show how to run the emotion models over the emotion dataset and save the results:

GRU model
``` bash
python run_model.py ../english_emotion_dataset/train.json ../english_emotion_dataset/development.json ../english_emotion_dataset/test.json ./training_configs/gru_emotion.json ./results/emotion/full_dataset/gru/
```
Output
```
Development averages:
jaccard_index: 0.5453821347952273
Macro_F1: 0.4531289873941704
Micro_F1: 0.6557846537799392

Test averages:
jaccard_index: 0.5460443606715469
Macro_F1: 0.45198515963343366
Micro_F1: 0.6541575608393904
```

CNN model
``` bash
python run_model.py ../english_emotion_dataset/train.json ../english_emotion_dataset/development.json ../english_emotion_dataset/test.json ./training_configs/cnn_emotion.json ./results/emotion/full_dataset/cnn/
```
Output
```
Development averages:
jaccard_index: 0.5413355906696764
Macro_F1: 0.46285220676989935
Micro_F1: 0.6533926340413673

Test averages:
jaccard_index: 0.5400201639416122
Macro_F1: 0.4609928538437342
Micro_F1: 0.6529417520121095
```

GRU with attention model
``` bash
python run_model.py ../english_emotion_dataset/train.json ../english_emotion_dataset/development.json ../english_emotion_dataset/test.json ./training_configs/attention_gru_emotion.json ./results/emotion/full_dataset/attention_gru/
```
Output
```
Development averages:
jaccard_index: 0.5435440180586908
Macro_F1: 0.4534931968517402
Micro_F1: 0.6524408200051153

Test averages:
jaccard_index: 0.5462883735881588
Macro_F1: 0.4558530083660029
Micro_F1: 0.6546885915948876
```

### SSL Part
We remove the following labels ``['anger' 'disgust' 'joy' 'love']``, these labels were chosen as `anger` and `joy` represent two of the largest labels that are not correlated within sentences and then the addition of two labels that are very to highly correlated without removing too many of the samples.

#### Creating the new standard datasets with 4 labels removed
We now create the training, development and test datasets that do not contain these labels as follows:
``` bash
python dataset_label_subset.py ../english_emotion_dataset/train.json ../english_emotion_dataset/temp_ssl_train.json anger disgust joy love
python dataset_label_subset.py ../english_emotion_dataset/development.json ../english_emotion_dataset/ssl_development.json anger disgust joy love
python dataset_label_subset.py ../english_emotion_dataset/test.json ../english_emotion_dataset/ssl_test.json anger disgust joy love
```
To see how the label distributions have changed we can run the following scripts:
``` bash
python json_stats.py ../english_emotion_dataset/temp_ssl_train.json
python json_stats.py ../english_emotion_dataset/ssl_development.json
python json_stats.py ../english_emotion_dataset/ssl_test.json
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/temp_ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json
```
Output
``` python
{'anticipation': 306, 'fear': 548, 'neutral': 204, 'optimism': 331, 'pessimism': 361, 'sadness': 665, 'surprise': 87, 'trust': 78}
Dataset size: 1549
{'anticipation': 31, 'fear': 52, 'neutral': 14, 'optimism': 35, 'pessimism': 27, 'sadness': 66, 'surprise': 7, 'trust': 8}
Dataset size: 148
{'anticipation': 127, 'fear': 219, 'neutral': 75, 'optimism': 151, 'pessimism': 147, 'sadness': 305, 'surprise': 30, 'trust': 38}
Dataset size: 628
{'anticipation': '20.0', 'fear': '35.2', 'neutral': '12.6', 'optimism': '22.2', 'pessimism': '23.0', 'sadness': '44.6', 'surprise': '5.3', 'trust': '5.3'}
```
#### Sub-Sampling the training dataset
Now that we have the new dataset splits we need to subsample the training dataset so that we can generate un-lablled examples and to treat it as a more low resource task. Therefore we are going to randomly sub-sample 50% of the training data which would make the development set ~25% of the size of the training data.
``` bash
python dataset_subset.py ../english_emotion_dataset/temp_ssl_train.json ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/unlablled_ssl_train.json 50
```

For the new SSL Training dataset
``` bash
python json_stats.py ../english_emotion_dataset/ssl_train.json
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/ssl_train.json
```
Output
``` python
{'anticipation': 146, 'fear': 255, 'neutral': 94, 'optimism': 149, 'pessimism': 177, 'sadness': 321, 'surprise': 36, 'trust': 33}
Dataset size: 737
{'anticipation': '19.8', 'fear': '34.6', 'neutral': '12.8', 'optimism': '20.2', 'pessimism': '24.0', 'sadness': '43.6', 'surprise': '4.9', 'trust': '4.5'}
```

For the unlablled data that has in distribution labels:
``` bash
python json_stats.py ../english_emotion_dataset/unlablled_ssl_train.json
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/unlablled_ssl_train.json
```
Output
``` python
{'anticipation': 160, 'fear': 293, 'neutral': 110, 'optimism': 182, 'pessimism': 184, 'sadness': 344, 'surprise': 51, 'trust': 45}
Dataset size: 812
{'anticipation': '19.7', 'fear': '36.1', 'neutral': '13.5', 'optimism': '22.4', 'pessimism': '22.7', 'sadness': '42.4', 'surprise': '6.3', 'trust': '5.5'}
```

#### Getting the unlablled out of label distribution data
For this we need to go back to the original data ``../english_emotion_dataset/train.json`` as this file is the only file that will contain samples with the out of label distribution data: ``['anger' 'disgust' 'joy' 'love']``. Therefore the following script will extract only samples that contain one or more of these labels and no other labels:
``` bash
python dataset_out_of_dist_subset.py ../english_emotion_dataset/train.json ../english_emotion_dataset/unlablled_ood_ssl_train.json anger disgust joy love
python json_stats.py ../english_emotion_dataset/unlablled_ood_ssl_train.json
python json_stats.py --normalise_by_sample_count ../english_emotion_dataset/unlablled_ood_ssl_train.json
```
Output
``` python
{'anger': 1076, 'disgust': 1016, 'joy': 552, 'love': 170}
Dataset size: 1590
{'anger': '67.7', 'disgust': '63.9', 'joy': '34.7', 'love': '10.7'}
```

#### Training the baseline models
Here we will not use any of the unlabelled data.

Attention GRU
``` bash
python run_model.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ./training_configs/attention_gru_emotion.json ./results/emotion/ssl/baselines/attention_gru
```

Output:
Development averages:
jaccard_index: 0.5019144144144144
Macro_F1: 0.37622917143261425
Micro_F1: 0.5612854791670319

Test averages:
jaccard_index: 0.47818471337579616
Macro_F1: 0.38128589668680496
Micro_F1: 0.5498059155977425

GRU
``` bash
python run_model.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ./training_configs/gru_emotion.json ./results/emotion/ssl/baselines/gru
```

Development averages:
jaccard_index: 0.4345045045045045
Macro_F1: 0.29492333525546494
Micro_F1: 0.48274794991122594

Test averages:
jaccard_index: 0.4236730360934182
Macro_F1: 0.3018631356074526
Micro_F1: 0.47508156934988105

CNN
``` bash
python run_model.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ./training_configs/cnn_emotion.json ./results/emotion/ssl/baselines/cnn
```

Output:
Development averages:
jaccard_index: 0.5193693693693694
Macro_F1: 0.3840422545993888
Micro_F1: 0.5774455915686852

Test averages:
jaccard_index: 0.49809447983014865
Macro_F1: 0.388864378605738
Micro_F1: 0.5726205032170439

Majority and Random baselines:

The majority class in this case is the majority class within the training dataset.
``` bash
python baseline_systems.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_test.json
```
Output
``` python
{'Majority jaccard index': 0.2918524416135881, 'Majority macro f1': 0.08172561629153269, 'Majority micro f1': 0.35465116279069764}
{'Random jaccard index': 0.1772520473157416, 'Random macro f1': 0.23813424563341148, 'Random micro f1': 0.2901139513396982}
```

#### Overlap between the systems in the predictions.
We now look at the overlap in the prediction between the 3 different trained models when trained on different bootstrap samples of the training data. 

We measure prediction overlap between two classifiers as the number of samples within the unlablled dataset that have the same prediction labels as well as the prediction overlap between all three classifiers. The following will look at the overlap in the whole of the **in** class distribution unlabelled data:
``` bash
python prediction_overlap.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_ssl_train.json ./training_configs/cnn_emotion.json ./training_configs/gru_emotion.json ./training_configs/attention_gru_emotion.json 
```
Output:
```
Development Scores:
Model cnn_emotion: {'jaccard_index': 0.535472972972973, 'Macro_F1': 0.37745629379114387, 'Micro_F1': 0.5862884160756501}
Model gru_emotion: {'jaccard_index': 0.4904279279279279, 'Macro_F1': 0.3347039072039072, 'Micro_F1': 0.5283950617283951}
Model attention_gru_emotion: {'jaccard_index': 0.48536036036036034, 'Macro_F1': 0.3323793835157472, 'Micro_F1': 0.5281173594132029}

Number of samples in unlabelled data: 812
Prediction overlaps:
cnn_emotion gru_emotion 464
cnn_emotion attention_gru_emotion 475
gru_emotion cnn_emotion 464
gru_emotion attention_gru_emotion 518
attention_gru_emotion cnn_emotion 475
attention_gru_emotion gru_emotion 518
Overlap between all three predictors: 359
```

And here we look at the overlap within the **out** of class distribution unlabelled data:
``` bash
python prediction_overlap.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_ood_ssl_train.json ./training_configs/cnn_emotion.json ./training_configs/gru_emotion.json ./training_configs/attention_gru_emotion.json
``` 
Output:
```
Development Scores:
Model cnn_emotion: {'jaccard_index': 0.5253378378378378, 'Macro_F1': 0.3741762630039079, 'Micro_F1': 0.5841121495327103}
Model gru_emotion: {'jaccard_index': 0.4115990990990991, 'Macro_F1': 0.28215935791235924, 'Micro_F1': 0.4720194647201947}
Model attention_gru_emotion: {'jaccard_index': 0.4594594594594595, 'Macro_F1': 0.32347102793529603, 'Micro_F1': 0.5}

Number of samples in unlabelled data: 1590
Prediction overlaps:
cnn_emotion gru_emotion 764
cnn_emotion attention_gru_emotion 842
gru_emotion cnn_emotion 764
gru_emotion attention_gru_emotion 666
attention_gru_emotion cnn_emotion 842
attention_gru_emotion gru_emotion 666
Overlap between all three predictors: 469
```

#### Tri-Training with disagreement
To ensure that we have the same amount of unlabelled data from in and out of class distribution data we need to sub-sample randomly the out of class distribution data:
``` bash
python dataset_subset_exact.py ../english_emotion_dataset/unlablled_ood_ssl_train.json ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json 812
python json_stats.py ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json
```
Output
``` python
{'anger': 543, 'disgust': 516, 'joy': 298, 'love': 89}
Dataset size: 812
```

Further more as using the original tri-training different archtectures did not work we are now sub-sampling the training data and training 3 CNN models. Therefore we need 3 subsets of the training data. The number of samples we have subseted is 80% randomly:
``` bash
python dataset_subset_exact.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_train_1.json 590
python dataset_subset_exact.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_train_2.json 590
python dataset_subset_exact.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_train_3.json 590
```
##### Tri-Training using the in-class distribution
``` bash
python tri_training.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_ssl_train.json ./results/emotion/ssl/another ./training_configs/cnn_emotion.json ./training_configs/attention_gru_emotion.json ./training_configs/gru_emotion.json
```

##### Tri-Training using 75% in 25% out class distribution
``` bash
python mix_datasets.py ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json ../english_emotion_dataset/unlablled_ssl_train.json ../english_emotion_dataset/unlablled_sub_ood_train_25.json 203 609

python tri_training.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_sub_ood_train_25.json ./results/emotion/ssl/s ./training_configs/cnn_emotion.json ./training_configs/attention_gru_emotion.json ./training_configs/gru_emotion.json
```

##### Tri-Training using 50% in 50% out class distribution
``` bash
python mix_datasets.py ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json ../english_emotion_dataset/unlablled_ssl_train.json ../english_emotion_dataset/unlablled_sub_ood_train_50.json 406 406

python tri_training.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_sub_ood_train_50.json ./results/emotion/ssl/day ./training_configs/cnn_emotion.json ./training_configs/attention_gru_emotion.json ./training_configs/gru_emotion.json
```

##### Tri-Training using 75% in 25% out class distribution
``` bash
python mix_datasets.py ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json ../english_emotion_dataset/unlablled_ssl_train.json ../english_emotion_dataset/unlablled_sub_ood_train_75.json 609 203
```


##### Tri-Training using the out-class distribution
``` bash
python tri_training.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json ./results/emotion/ssl/anything ./training_configs/cnn_emotion.json ./training_configs/attention_gru_emotion.json ./training_configs/gru_emotion.json
```


python tri_training.py ../english_emotion_dataset/ssl_train.json ../english_emotion_dataset/ssl_development.json ../english_emotion_dataset/ssl_test.json ../english_emotion_dataset/unlablled_subset_ood_ssl_train.json ./results/emotion/ssl/anything ./training_configs/cnn_emotion.json ./training_configs/attention_gru_emotion.json ./training_configs/gru_emotion.json
{'jaccard_index': [0.490427927927928, 0.4960585585585585, 0.5033783783783784, 0.5033783783783784, 0.5275900900900901]}
{'jaccard_index': [0.4612526539278131, 0.4986730360934183, 0.4865711252653927, 0.49004777070063693, 0.5013800424628451]}

```
python plot.py
```

## Format the OntoNotes 5.0 data
The data can be found from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19). The instructions on how to format it can be found [here](https://cemantix.org/data/ontonotes.html) however these instructions are only valid up to step 3 due to the script not being avaliable from step. Therefore from step 3 onwards the scripts to download can be found [here](http://conll.cemantix.org/2012/data.html). From this you need to do the following:
1. Within `conll-formatted-ontonotes-5.0-12` folder create the following two directories `conll-2012` and `v4` and then put the `v4` folder within `conll-2012`.
2. Put the `data` folder that is within `conll-formatted-ontonotes-5.0-12` in the `v4` subdirectory.
3. Put the `scripts` folder that you downloaded from [here](http://conll.cemantix.org/2012/data.html) into the `v4` folder also.
4. At this stage the `conll-formatted-ontonotes-5.0-12` should look like this `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/data` and `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/scripts`
5. Assuming that the `ontonotes-release-5.0` folder downloaded from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) is in the same directory as `conll-formatted-ontonotes-5.0-12` run the following command from the `scripts` folder: `./skeleton2conll.sh -D ../../../../ontonotes-release-5.0/data/files/data/ ../../../conll-2012/`
6. That last command should have added the annotations from the ontonotes 5.0 data for English using the corrected text files from [here](https://github.com/ontonotes/conll-formatted-ontonotes-5.0/releases/tag/v12).

