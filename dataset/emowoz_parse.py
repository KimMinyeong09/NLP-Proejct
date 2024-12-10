import json 
import pandas as pd 

out_path = 'test'

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
                emos.append(-1) # system turns 
            else: 
                emos.append(v['emotion'][3]['emotion'])

        seq_len = len(seqs)
        emo_len = len(emos)

        if seq_len < 6: 
            continue

        # Get rid of the blanks at the start & end of each turns
        seqs = [s.strip() for s in seqs]
        seqs = [s.replace(' .', '.') for s in seqs]
        seqs = [s.replace(' ?', '?') for s in seqs]
        seqs = [s.replace(' !', '!') for s in seqs]
        seqs = [s.replace(' ,', ',') for s in seqs]
        seqs = [s.replace(" ’", "’") for s in seqs]

        seq_count = seq_len // 5

        for index in range(0, 5 * seq_count, 5): 
            # global 
            global_seq.append(' '.join(seqs[index : index + 4]))
            global_emo.append(emos[index + 3]) 

            # local
            local_seq.append(seqs[index + 4])
            local_emo.append(emos[index + 4])

data_dic = {
    'global_seq': global_seq,
    'global_emo': global_emo, 
    'local_seq': local_seq, 
    'local_emo': local_emo}

df = pd.DataFrame(data_dic)
df.to_csv(out_path + ".csv", encoding = 'utf-8')
