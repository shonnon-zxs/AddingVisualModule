import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
import random

def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4



class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, normal, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.normal = normal
        self.classifier = classifier
        self.debias_loss_fn = None

        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, q, labels, bias,v_mask, qtype):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att = self.v_att(v, q_emb)
        batch_size = q.size(0)
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)
        v_emb = (att * v).sum(1)  # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        # construct an irrelevant Q-I pair for each instance
        index = random.sample(range(0, batch_size), batch_size)
        gv_neg = v[index]
        logits_neg, att_gv_neg, joint = \
            self.compute_predict(q_repr, q_emb, gv_neg)
        # return logits_pos, logits_neg, att_gv_pos, att_gv_neg, joint

        if labels is not None:
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
        else:
            loss = None

        return logits, logits_neg, loss, w_emb

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.v_att(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.v_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        return logits, att_gv, joint_repr_normal

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    normal = nn.BatchNorm1d(num_hid, affine=False)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, normal, classifier)