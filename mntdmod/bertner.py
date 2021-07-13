# -*- coding: utf-8 -*-

"""
bertner.py
~~~~~~~~~~

This is a replication of NIST's TrojAI Round 7 NER classifier.

"""
import torch
from torch import nn
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NerLinearModel(nn.Module):
    """Reimplementation of the NIST NerLinearModel.

    NIST's Round 7 TrojAI task uses pretrained 🤗 BERT models for the embeddings
    and add a simple dropout and linear classification trailer for the NER task itself.

    Details of the NIST architecture can be found at:
        * Documentation: https://pages.nist.gov/trojai/docs/data.html#round-7
        * Saved models: https://data.nist.gov/od/id/mds2-2407

    Attributes:
        embedding_flavor: The BERT embedding architecture to use, can be one of
            ['bert-base-uncased' | 'distilbert-base-cased' | 'google/mobilebert-uncased' | 'roberta-base']
            for the BERT, DistilBERT, MobileBERT, and RoBERTa variant architectures, respectively.
        num_labels: The number of output labels for the dataset.
        ignore_index: Tokens with this label are ignored during loss calculation.
        transformer: The pretrained BERT architecture from 🤗.
        dropout: A dropout layer to prevent overfitting (NIST uses 10% dropout).
        classifier: A final linear trailer classifier for the NER task.

    """

    def __init__(self, embedding_flavor="bert-base-uncased", num_labels=9):
        super().__init__()

        embedding_size = 512 if embedding_flavor == "google/mobilebert-uncased" else 768

        self.embedding_flavor = embedding_flavor
        self.num_labels = num_labels
        self.ignore_index = -100

        self.transformer = BertModel.from_pretrained(self.embedding_flavor)
        #  New trailer layers to match NIST implementation
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_size, self.num_labels, bias=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

        return loss, emissions


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = CustomBERTModel()  # You can pass the parameters if required to have more flexible model
# model.to(torch.device(device))  ## can be gpu
# criterion = nn.CrossEntropyLoss()  ## If required define your own criterion
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
#
# for epoch in epochs:
#     for batch in data_loader:  ## If you have a DataLoader()  object to get the data.
#
#         data = batch[0]
#         targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets
#
#         optimizer.zero_grad()
#         encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
#                                                add_special_tokens=True)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         outputs = F.log_softmax(outputs, dim=1)
#         input_ids = encoding['input_ids']
#         attention_mask = encoding['attention_mask']
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
