from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import ModelOutput
from utils import get_label_list


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MyTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: dict[str, Optional[torch.FloatTensor]] = None
    logits: dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MyBertForTokenClassification(nn.Module):
    def __init__(self,
                 path,
                 max_seq_len: int = 128,
                 num_labels: int = 3,
                 num_heads: int = 1):
        super().__init__()
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bert = BertModel.from_pretrained(path, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)
                active_labels = torch.where(active_loss, labels.view(-1),
                                            torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassification2Heads(nn.Module):
    def __init__(self,
                 path,
                 max_seq_len: int = 128,
                 num_labels: int = 3,
                 num_heads: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bert = BertModel.from_pretrained(path, add_pooling_layer=False)
        self.dropout_topic = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_topic = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout_target = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_target = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        sequence_output_topic = self.dropout_topic(sequence_output)
        logits_topic = self.classifier_topic(sequence_output_topic)
        sequence_output_target = self.dropout_target(sequence_output)
        logits_target = self.classifier_target(sequence_output_target)
        logits_dict = {'topic': logits_topic,
                       'target': logits_target}
        loss_dict = {'topic': None,
                     'target': None}
        label_dict = {'topic': labels[:, 0],
                      'target': labels[:, 1]}
        for k in logits_dict.keys():
            logits = logits_dict[k]
            label = label_dict[k]
            if label is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(active_loss, label.reshape(-1),
                                                torch.tensor(loss_fct.ignore_index).type_as(label))
                    loss_dict[k] = loss_fct(active_logits, active_labels)
                else:
                    loss_dict[k] = loss_fct(logits.view(-1, self.num_labels), label.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return MyTokenClassifierOutput(
            loss=loss_dict,
            logits=logits_dict,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
