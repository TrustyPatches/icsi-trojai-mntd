# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch
import trojai
from trojai.modelgen.architecture_factory import ArchitectureFactory

ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'Linear']

class NerLinearModel(torch.nn.Module):
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
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

        return loss, emissions


class LinearModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # get rid of implicit sequence length
        # for GRU and LSTM input needs to be [batch size, sequence length, embedding length]
        # sequence length is 1
        # however the linear model need the input to be [batch size, embedding length]
        data = data[:, 0, :]
        # input data is after the embedding
        hidden = self.dropout(data)

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class GruLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.GRU(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        _, hidden = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class LstmLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class LinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LinearModel(input_size, output_size, dropout)
        return model


class GruLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = GruLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model


class LstmLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LstmLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model