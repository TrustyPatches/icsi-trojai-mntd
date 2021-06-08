import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaClassifier(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, N_tokens=1, gpu=False):
        super(MetaClassifier, self).__init__()
        self.input_size = input_size    # Size of original raw data input
        self.class_num = class_num      # Size of target model logit vector
        self.N_in = N_in                # K query sets
        self.N_h = 20                   # Size of hidden layer (metaclassifier capacity)?
        self.N_tokens = N_tokens        # Size of input tokens for NER task   # For TrojAI

        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_() * 1e-3)    # This layer is the query set
        # --------------v Slicing here leaves just the metaclassifier  v---------------- #
        self.fc = nn.Linear(self.N_in * self.class_num * self.N_tokens, self.N_h)        # Main metaclassifier layer
        self.output = nn.Linear(self.N_h, 1)                                             # Output layer

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred):
        # import IPython; IPython.embed(); exit()
        emb = F.relu(self.fc(pred.view(self.N_in * self.class_num * self.N_tokens)))
        score = self.output(emb)
        return score

    def loss(self, score, y):
        y_var = torch.FloatTensor([y])
        if self.gpu:
            y_var = y_var.cuda()
        l = F.binary_cross_entropy_with_logits(score, y_var)
        return l


class MetaClassifierOC(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, gpu=False):
        super(MetaClassifierOC, self).__init__()
        self.N_in = N_in
        self.N_h = 20
        self.v = 0.1
        self.input_size = input_size
        self.class_num = class_num

        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_()*1e-3)
        self.fc = nn.Linear(self.N_in*self.class_num, self.N_h)
        self.w = nn.Parameter(torch.zeros(self.N_h).normal_()*1e-3)
        self.r = 1.0

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred, ret_feature=False):
        emb = F.relu(self.fc(pred.view(self.N_in*self.class_num)))
        if ret_feature:
            return emb
        score = torch.dot(emb, self.w)
        return score

    def loss(self, score):
        reg = (self.w**2).sum()/2
        for p in self.fc.parameters():
            reg = reg + (p**2).sum()/2
        hinge_loss = F.relu(self.r - score)
        loss = reg + hinge_loss / self.v - self.r
        return loss

    def update_r(self, scores):
        self.r = np.asscalar(np.percentile(scores, 100*self.v))
        return
