import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, data=''):
        self.batch_size = 512
        self.ptr = 0

        config_file = '../data/' + data + '/config.txt'
        with open('../data/' + data + '/probability.json', encoding='utf8') as i_f:
            school_A = json.load(i_f)
        with open('../data/' + data + '/function_val.json', encoding='utf8') as i_f:
            available_school_B = json.load(i_f)
        self.data = school_A + available_school_B
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = map(eval, i_f.readline().split(','))
        self.knowledge_dim = int(knowledge_n)

        with open('../data/' + data + '/user_id.json', encoding='utf8') as i_f:
            self.user_id = json.load(i_f)
        with open('../data/' + data + '/exercise_id.json', encoding='utf8') as i_f:
            self.exercise_id = json.load(i_f)
        with open('../data/' + data + '/knowledge_id.json', encoding='utf8') as i_f:
            self.knowledge_id = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                knowledge_emb[self.knowledge_id.index(knowledge_code)] = 1.0
            y = log['score']
            input_stu_ids.append(self.user_id.index(log['user_id']))
            input_exer_ids.append(self.exercise_id.index(log['exer_id']))
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, data='', d_type='validation'):
        self.batch_size = 512
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            data_file = '../data/' + data + '/probability_val.json'
        else:
            data_file = '../data/' + data + '/function.json'
        config_file = '../data/' + data + '/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = map(eval, i_f.readline().split(','))
            self.knowledge_dim = int(knowledge_n)
        with open('../data/' + data + '/user_id.json', encoding='utf8') as i_f:
            self.user_id = json.load(i_f)
        with open('../data/' + data + '/exercise_id.json', encoding='utf8') as i_f:
            self.exercise_id = json.load(i_f)
        with open('../data/' + data + '/knowledge_id.json', encoding='utf8') as i_f:
            self.knowledge_id = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                knowledge_emb[self.knowledge_id.index(knowledge_code)] = 1.0
            y = log['score']
            input_stu_ids.append(self.user_id.index(log['user_id']))
            input_exer_ids.append(self.exercise_id.index(log['exer_id']))
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0