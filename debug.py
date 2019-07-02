import json
import shutil
import sys
import tempfile

from allennlp.commands import main

config_file = "./training_configs/test_predictor_config.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

with tempfile.TemporaryDirectory() as serialization_dir:

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "nlp_uncertainty_ssl"
    ]

    main()