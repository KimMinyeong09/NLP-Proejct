import json 
import pandas as pd 

out_path = 'train'

with open('emowoz-multiwoz.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

with open('dataset_split.json', 'r', encoding='utf-8') as f:
    dataset_id = json.load(f)

dataset_id = dataset_id[out_path]['multiwoz']

global_seq = [] 
local_seq = [] 
global_emo = [] 
local_emo = []

for key, value in loaded_data.items():

    if key in dataset_id: 
        value = value['log']

        # 전체 text와 emotion을 하나의 리스트에 넣기 
        seqs = [] 
        emos = [] 

        for i, v in enumerate(value): 
            seqs.append(v['text'].strip())
            
            if i % 2 == 1: 
                emos.append(7) # system turns 
            else: 
                emos.append(v['emotion'][3]['emotion'])

        seq_len = len(seqs)
        emo_len = len(emos)

        # Get rid of the blanks at the start & end of each turns
        # seqs = [s.strip() for s in seqs]
        seqs = [s.replace(' .', '.') for s in seqs]
        seqs = [s.replace(' ?', '?') for s in seqs]
        seqs = [s.replace(' !', '!') for s in seqs]
        seqs = [s.replace(' ,', ',') for s in seqs]
        seqs = [s.replace(" ’", "’") for s in seqs]

        if seq_len < 3: continue # 길이가 3이상인 발화에 대하여

        for index in range(1, seq_len - 1): # index seq
            # global_seq에 스페셜 토큰 추가 
            seqs_three = [seqs[i].strip() for i in range(index - 1, index + 2)]
            global_index_seq = "<s> " + " </s> ".join(seqs_three) + " </s>"
            global_index_emo = emos[index]
            
            for index_ in [index - 1, index + 1]: # pre: (index - 1) post: (index + 1)
                seqs[index_] = seqs[index_].strip()
                local_index_seq = seqs[index_]
                local_index_emo = emos[index_]

                global_seq.append(global_index_seq)
                global_emo.append(global_index_emo)

                local_seq.append(local_index_seq)
                local_emo.append(local_index_emo)

data_dic = {
    'global_seq': global_seq,
    'global_emo': global_emo, 
    'local_seq': local_seq, 
    'local_emo': local_emo}

df = pd.DataFrame(data_dic)
df.to_csv(out_path + ".csv", encoding = 'utf-8')
