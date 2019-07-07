To created the following saved models run the associated command at the top of this project directory.

cvt_tagger -- `allennlp train training_configs/test_predictor_config.json -s tests/predictors/saved_models/cvt_tagger --include-package nlp_uncertainty_ssl`

emotion_classifier -- `allennlp train training_configs/test_emotion_predictor_config.json -s ./tests/predictors/saved_models/emotion_classifier --include-package nlp_uncertainty_ssl`