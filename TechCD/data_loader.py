import json
import torch
import random

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, data='', source='', target_train=''):
        self.batch_size = 256
        self.ptr = 0
        self.data = []

        with open('../data/' + data + '/config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = map(eval, i_f.readline().split(','))
            self.knowledge_dim = self.knowledge_n
        with open('../data/' + data + source, encoding='utf8') as i_f:
            school_A = json.load(i_f)
        with open('../data/' + data + target_train, encoding='utf8') as i_f:
            available_school_B = json.load(i_f)
        self.data = school_A + available_school_B

        with open('../data/' + data + '/user_id.json', encoding='utf8') as i_f:
            self.user_id = json.load(i_f)
        with open('../data/' + data + '/exercise_id.json', encoding='utf8') as i_f:
            self.exercise_id = json.load(i_f)
        with open('../data/' + data + '/knowledge_id.json', encoding='utf8') as i_f:
            self.knowledge_id = json.load(i_f)
        with open('../data/' + data + '/history.json', encoding='utf8') as i_f:
            self.history_data = json.load(i_f)
        with open('../data/' + data + '/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_exer_emb, input_knowedge_embs, input_knowedge_ids, ys, history = [], [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                knowledge_emb[self.knowledge_id.index(knowledge_code)] = 1.0
            input_knowedge_ids.append(self.knowledge_id.index(log['knowledge_code']))
            y = log['score']
            input_stu_ids.append(self.user_id.index(log['user_id']))
            input_exer_ids.append(self.exercise_id.index(log['exer_id']))
            input_exer_emb.append(log['exer_name'])
            # input_exer_emb.append(self.exer_embedding[log['exer_name']][0])
            exer_name = log['exer_name']
            if exer_name in self.exer_embedding:
                input_exer_emb.append(self.exer_embedding[exer_name][0])  # Add the embedding itself, not the name
            else:
                print(f"Warning: Exercise name {exer_name} not found in exer_embedding")
            # print (self.embedding[log['exer_id']])
            # print(type(self.exer_embedding[exer_name][0]))
            input_knowedge_embs.append(knowledge_emb)
            if str(self.user_id.index(log['user_id'])) not in self.history_data.keys():
                history.append([random.randint(0, self.exer_n-1) for _ in range(20)])
            else:
                history.append(self.history_data[str(self.user_id.index(log['user_id']))])
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(input_knowedge_ids), torch.LongTensor(ys), history, input_exer_emb

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation', data='', source_val='', target=''):
        self.batch_size = 256
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        with open('../data/' + data + '/iflytek_exercise_embedding.json', encoding='utf8') as i_f:
            self.exer_embedding = json.load(i_f)

        with open('../data/' + data + '/history.json', encoding='utf8') as i_f:
            self.history_data = json.load(i_f)
        if d_type =='validation':
            data_file = '../data/' + data + source_val
        else:
            data_file = '../data/' + data + target

        config_file = '../data/' + data + '/config.txt'

        with open('../data/' + data + '/user_id.json', encoding='utf8') as i_f:
            self.user_id = json.load(i_f)
        with open('../data/' + data + '/exercise_id.json', encoding='utf8') as i_f:
            self.exercise_id = json.load(i_f)
        with open('../data/' + data + '/knowledge_id.json', encoding='utf8') as i_f:
            self.knowledge_id = json.load(i_f)

        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = map(int, i_f.readline().split(','))
            self.knowledge_dim = self.knowledge_n

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_exer_emb, input_knowedge_embs, input_knowedge_ids, ys, history = [], [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in [log['knowledge_code']]:
                knowledge_emb[self.knowledge_id.index(knowledge_code)] = 1.0
            input_knowedge_ids.append(self.knowledge_id.index(log['knowledge_code']))
            y = log['score']
            input_stu_ids.append(self.user_id.index(log['user_id']))
            input_exer_ids.append(self.exercise_id.index(log['exer_id']))
            input_exer_emb.append(self.exer_embedding[log['exer_name']][0])
            input_knowedge_embs.append(knowledge_emb)
            # print (self.embedding[log['exer_id']])
            if str(log['user_id']) not in self.history_data.keys():
                history.append([random.randint(0, self.exer_n-1) for _ in range(20)])
            else:
                history.append(self.history_data[str(self.user_id.index(log['user_id']))])
            ys.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(input_knowedge_ids), torch.LongTensor(ys), history, input_exer_emb

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
