import torch
import torch.nn as nn


class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, 1)
        self.e_discrimination = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = self.student_emb(stu_id)
        k_difficulty = self.k_difficulty(exer_id)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 5
        # prednet
        input_x = torch.sum(e_discrimination * stu_emb, dim=1, keepdim=True) + k_difficulty
        # input_x = self.prednet_full1(input_x)
        # input_x = self.prednet_full2(input_x)
        output = torch.sigmoid(input_x)
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
