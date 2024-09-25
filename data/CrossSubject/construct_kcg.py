import os
import json
import numpy as np
from collections import defaultdict

exer_n = 0
knowledge_n = 0
student_n = 0

def merge_train_data():
    with open('./function.json') as f:
        function = json.load(f)
    with open('./function_val.json') as f:
        function_val = json.load(f)
    with open('./probability.json') as f:
        probability = json.load(f)
    with open('./probability_val.json') as f:
        probability_val = json.load(f)
    # data = function + function_val

    # print(len(data))

    all_data = function + function_val + probability + probability_val
    # data = all_data
    user_id = []
    knowledge_id = []
    exercise_id = []
    exer_concept = []
    for log in all_data:
        user_id.append(log['user_id'])
        for k in [log['knowledge_code']]:
            knowledge_id.append(k)
            exer_concept.append((log['exer_id'], k))
        exercise_id.append(log['exer_id'])
    user_id = list(set(user_id))
    knowledge_id = list(set(knowledge_id))
    exercise_id = list(set(exercise_id))
    user_id.sort()
    knowledge_id.sort()
    exercise_id.sort()
    with open('./user_id.json', 'w') as f:
        json.dump(user_id, f, indent=4, ensure_ascii=False)
    with open('./knowledge_id.json', 'w') as f:
        json.dump(knowledge_id, f, indent=4, ensure_ascii=False)
    with open('./exercise_id.json', 'w') as f:
        json.dump(exercise_id, f, indent=4, ensure_ascii=False)
    # print (exercise_id)
    print (len(user_id))
    with open('config.txt', 'w')as f:
        f.write('# Number of Students, Number of Exercises, Number of Knowledge Concepts\n'+str(len(user_id))+','+str(len(exercise_id))+','+str(len(knowledge_id)))

    exer_concept = list(set(exer_concept))
    exer_concept_str = ''
    for exer_c in exer_concept:
        exer_concept_str += str(exercise_id.index(exer_c[0])) + '\t' + str(knowledge_id.index(exer_c[1])) + '\n'
    with open('./Exer_Concept.txt', 'w') as f:
        f.write(exer_concept_str)

    # 用于存储分类后的数据
    classified_data = defaultdict(lambda: {"log_num": 0, "logs": []})

    # 遍历原始数据并进行分类
    for entry in all_data:
        # print(entry)
        u_id = entry["user_id"]
        log_entry = {
            "exer_id": entry["exer_id"],
            "score": entry["score"],
            "knowledge_code": entry["knowledge_code"]
        }
        classified_data[u_id]["logs"].append(log_entry)
        classified_data[u_id]["log_num"] += 1

    # 将分类数据转换为所需的列表格式
    result = [{"user_id": u_id, **details} for u_id, details in classified_data.items()]
    print (len(result))

    # 输出结果
    with open('./train_log.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    data = function + function_val + probability_val
    # 遍历原始数据并进行分类
    for entry in data:
        # print(entry)
        u_id = entry["user_id"]
        log_entry = {
            "exer_id": entry["exer_id"],
            "score": entry["score"],
            "knowledge_code": entry["knowledge_code"]
        }
        classified_data[u_id]["logs"].append(log_entry)
        classified_data[u_id]["log_num"] += 1

    # 将分类数据转换为所需的列表格式
    data_result = [{"user_id": u_id, **details} for u_id, details in classified_data.items()]
    # print(len(data_result))

    print (len(exercise_id))

    history = {}
    for user in data_result:
        temp_user_history = []
        for l in user['logs']:
            temp_user_history.append(exercise_id.index(l['exer_id']))
        history[str(user_id.index(user['user_id']))] = temp_user_history
    with open('./history.json', 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def construct_dependency_matrix():
    # 判断文件是否存在
    file_path = './config.txt'
    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
    else:
        print(f"The file '{file_path}' does not exist.")
        merge_train_data()
    with open('./train_log.json') as f:
        data = json.load(f)
    with open('./config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = map(int, i_f.readline().split(','))

    with open('./knowledge_id.json') as f:
        knowledge_id = json.load(f)


    # Initialize matrices and dictionaries
    knowledge_correct = np.zeros((knowledge_n, knowledge_n))
    edge_dic_deno = {}

    # Calculate correct matrix
    for student in data:
        if student['log_num'] < 2:
            continue
        logs = student['logs']
        for log_i in range(student['log_num'] - 1):
            if logs[log_i]['score'] * logs[log_i + 1]['score'] == 1:
                for ki in [logs[log_i]['knowledge_code']]:
                    for kj in [logs[log_i + 1]['knowledge_code']]:
                        if ki != kj:
                            knowledge_correct[knowledge_id.index(ki), knowledge_id.index(kj)] += 1.0
                            edge_dic_deno[knowledge_id.index(ki)] = edge_dic_deno.get(knowledge_id.index(ki), 0) + 1

    # Calculate transition matrix
    knowledge_directed = np.zeros((knowledge_n, knowledge_n))
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if i != j and knowledge_correct[i, j] > 0:
                knowledge_directed[i, j] = knowledge_correct[i, j] / edge_dic_deno[i]

    # Normalize transition matrix
    min_c = np.min(knowledge_directed[knowledge_directed > 0])
    max_c = np.max(knowledge_directed)
    normalized_o = (knowledge_directed - min_c) / (max_c - min_c)

    # Calculate threshold and filter edges
    threshold = np.power(np.mean(normalized_o[normalized_o > 0]), 4)
    threshold = 0.1
    edges = normalized_o >= threshold
    graph_edges = [(i, j) for i in range(knowledge_n) for j in range(knowledge_n) if edges[i, j]]

    print(len(graph_edges))

    # Write graph to file
    with open('knowledgeGraph.txt', 'w') as f:
        f.writelines(f"{i}\t{j}\n" for i, j in graph_edges)

    # Calculate directed and undirected edges
    directed_edges, undirected_edges = set(), set()
    for src, tar in graph_edges:
        if (tar, src) in graph_edges:
            undirected_edges.add((src, tar))
        else:
            directed_edges.add((src, tar))

    # Write directed and undirected graphs to files
    with open('K_Directed.txt', 'w') as f:
        f.writelines(f"{i}\t{j}\n" for i, j in directed_edges)
    with open('K_Undirected.txt', 'w') as f:
        f.writelines(f"{i}\t{j}\n" for i, j in undirected_edges)

if __name__ == '__main__':
    construct_dependency_matrix()
