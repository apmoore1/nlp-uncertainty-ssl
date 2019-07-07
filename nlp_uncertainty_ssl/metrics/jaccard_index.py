from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

from nlp_uncertainty_ssl.emotion_metrics import jaccard_index


@Metric.register("jaccard_index")
class JaccardIndex(Metric):
    """
    Performs Jaccard Index: 
    :math:`\frac{1}{N} \sum_{n=1}^{N} \frac{G_n \cap P_n}{G_n \cup P_n}`
    
    :param incl_neutral: Whether to include the neutral label in the score 
                         where the neutral label is denoted by all the lables 
                         being 0
    """
    def __init__(self, incl_neutral: bool) -> None:
        self.incl_neutral = incl_neutral
        self.jaccard_score = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, num_classes). It must be the same
            shape as the ``predictions``.
        :raises ConfigurationError: If the size of the predictions and gold labels 
                                    are not equall.
        """
        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)

        # Some sanity checks.
        if gold_labels.dim() != predictions.dim():
            raise ConfigurationError("gold_labels must have dimension == predictions.size() but "
                                     f"found tensor of shape (predictions): {predictions.size()}"
                                     f"\nFound tensor of shape (gold): {gold_labels.size()}")
        if gold_labels.shape != predictions.shape:
            raise ConfigurationError("The gold labels are not of the same shape as predictions:\n"
                                     f"Predictions: {predictions.shape}\nGold: {gold_labels.shape}")
        predictions = predictions.numpy()
        gold_labels = gold_labels.numpy()
        mean_jaccard_score = jaccard_index(predictions, gold_labels, 
                                           incl_neutral=self.incl_neutral)
        batch_size = predictions.shape[0]
        jaccard_score = mean_jaccard_score * batch_size

        self.total_count += batch_size
        self.jaccard_score += jaccard_score

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated Jaccard Index.
        """
        if self.total_count > 1e-12:
            jaccard_score = float(self.jaccard_score) / float(self.total_count)
        else:
            jaccard_score = 0.0
        if reset:
            self.reset()
        return jaccard_score

    @overrides
    def reset(self):
        self.jaccard_score = 0.0
        self.total_count = 0.0