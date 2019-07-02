# nlp-uncertainty-ssl

## Previous work
Current SOTA for NLP and SSL is [Cross-View Training](https://www.aclweb.org/anthology/D18-1217) what we could do is re-implement this and only use the samples that the main model is very confident to train the auxilary predictors.


## Format the OntoNotes 5.0 data
The data can be found from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19). The instructions on how to format it can be found [here](https://cemantix.org/data/ontonotes.html) however these instructions are only valid up to step 3 due to the script not being avaliable from step. Therefore from step 3 onwards the scripts to download can be found [here](http://conll.cemantix.org/2012/data.html). From this you need to do the following:
1. Within `conll-formatted-ontonotes-5.0-12` folder create the following two directories `conll-2012` and `v4` and then put the `v4` folder within `conll-2012`.
2. Put the `data` folder that is within `conll-formatted-ontonotes-5.0-12` in the `v4` subdirectory.
3. Put the `scripts` folder that you downloaded from [here](http://conll.cemantix.org/2012/data.html) into the `v4` folder also.
4. At this stage the `conll-formatted-ontonotes-5.0-12` should look like this `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/data` and `/conll-formatted-ontonotes-5.0-12/conll-2012/v4/scripts`
5. Assuming that the `ontonotes-release-5.0` folder downloaded from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) is in the same directory as `conll-formatted-ontonotes-5.0-12` run the following command from the `scripts` folder: `./skeleton2conll.sh -D ../../../../ontonotes-release-5.0/data/files/data/ ../../../conll-2012/`
6. That last command should have added the annotations from the ontonotes 5.0 data for English using the corrected text files from [here](https://github.com/ontonotes/conll-formatted-ontonotes-5.0/releases/tag/v12).

