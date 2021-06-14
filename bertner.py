from transformers import BertModel, AutoTokenizer
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomBERTModel(nn.Module):

    def __init__(self, num_labels=9):
        super().__init__()

        self.num_labels = num_labels
        self.ignore_index = -100

        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        ### New layers:
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(768, self.num_labels, bias=True)


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

            if attention_mask is not None:
                try:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = emissions.view(-1, self.num_labels)
                    active_labels = torch.where(active_loss, labels.view(-1),
                                                torch.tensor(loss_fct.ignore_index).type_as(labels))
                    loss = loss_fct(active_logits, active_labels)
                except:
                    import IPython; IPython.embed(); exit()
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

        return loss, emissions

#
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