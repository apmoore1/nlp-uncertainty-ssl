from typing import Dict, Optional, List, Any

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from nlp_uncertainty_ssl.metrics.jaccard_index import JaccardIndex

@Model.register("emotion_classifier")
class EmotionClassifier(Model):
    """
    The ``emotion_classifier`` is a multi label classifier (predict 0-N labels per
    sample).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``, optional (default=None)
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1. 
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``). Use `Variational Dropout 
              <https://arxiv.org/abs/1512.05287>`_ for sequence and normal 
              dropout for non sequences.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 label_namespace: str = "labels",
                 encoder: Optional[Seq2VecEncoder] = None,
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 incl_neutral: Optional[bool] = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_labels = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder

        embedding_output_dim = self.text_field_embedder.get_output_dim()
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
            self.variational_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        elif encoder is not None:
            output_dim = self.encoder.get_output_dim()
        else:
            output_dim = embedding_output_dim
        # Have to create a tag projection layer for each label in the 
        # multi label classifier
        self._tag_projection_layers: Any = []
        for k in range(self.num_labels):
            tag_projection_layer = Linear(output_dim, 1)
            self.add_module(f'tag_projection_layer_{k}', tag_projection_layer)
            self._tag_projection_layers.append(tag_projection_layer)
        self.output_activation = torch.nn.Sigmoid()
        self.loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        self.incl_neutral = incl_neutral
        self.metrics = {"jaccard_index": JaccardIndex(self.incl_neutral)}
        if encoder is not None:
            check_dimensions_match(embedding_output_dim, encoder.get_input_dim(),
                                   "text field embedding dim", "encoder input dim")
        if feedforward is not None and encoder is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        elif feedforward is not None and encoder is None:
            check_dimensions_match(embedding_output_dim, feedforward.get_input_dim(),
                                   "text field output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        labels : ``torch.LongTensor``, optional (default = ``None``)
                 A torch tensor representing the multiple labels that the sample 
                 can be as a one hot vector where each True label is 1 and the 
                 rest 0.
                 ``(batch_size, num_labels)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg:

            1. ``text`` - Original sentence
            2. ``words`` - Tokenised words from the sentence
            3. ``ID`` - Optionally the ID of the sample

        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.FloatTensor``
            The logits that are the output of the ``N`` tag projection layers 
            where each projection layer represents a different tag.
        probs: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_labels)`` 
            The probability that the sample is one of those labels. > 0.5 
            suggests that a label is associated to that sample.
        labels : ``List[List[int]]``
            The predicted labels where the inner list represents the multi label 
            classification.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``labels`` are provided.
        words : ``List[List[str]]``
            The tokens that were given as input
        text: ``List[str]``
            The text that was given to the tokeniser.
        ID: ``List[str]``
            The ID that is associated to the training example. Only returned if the ``ID`` are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        encoded_text = embedded_text_input

        if self.dropout is not None:
            encoded_text = self.variational_dropout(encoded_text)
        
        if self.encoder is not None: 
            encoded_text = self.encoder(encoded_text, mask)
            if self.dropout is not None:
                encoded_text = self.dropout(encoded_text)

        # Dropout is applied after each layer for feed forward if specified 
        # in the config.
        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        batch_size = embedded_text_input.shape[0]
        all_label_logits = torch.empty(batch_size, self.num_labels)
        for i in range(len(self._tag_projection_layers)):
            tag_projection = getattr(self, f'tag_projection_layer_{i}')
            i_tag_predictions = tag_projection(encoded_text).reshape(-1)
            all_label_logits[:, i] = i_tag_predictions
        probs = self.output_activation(all_label_logits)
        predicted_labels = probs > 0.5
        output = {'probs': probs, 'logits': all_label_logits,
                  'labels': predicted_labels}

        if labels is not None:
            labels = labels.type(torch.FloatTensor)
            loss = self.loss_criterion(all_label_logits, labels)
            output["loss"] = loss

            for metric in self.metrics.values():
                metric(predicted_labels, labels)

        if metadata is not None:
            words, texts, ids = [], [], []
            for sample in metadata:
                words.append(sample['words'])
                texts.append(sample['text'])
                if 'ID' in sample:
                    ids.append(sample['ID'])
            output["words"] = words
            output["text"] = texts
            if ids:
                output['ID'] = ids
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the labels to the actual labels. ``output_dict["readable_labels"]``
        is a list of lists which will contain zero or more readable labels.
        
        The type associated to the value of ``output_dict["readable_labels"]`` is 
        List[List[str]].
        """
        readable_labels: List[List[str]] = []
        for sample in output_dict['labels']:
            sample_labels: List[str] = []
            sample: List[int]
            # This should be a list of 0's and 1's
            for index, multi_label in enumerate(sample):
                if multi_label:
                    word_label = self.vocab.get_token_from_index(index, namespace=self.label_namespace)
                    sample_labels.append(word_label)
            readable_labels.append(sample_labels)
        output_dict['readable_labels'] = readable_labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return