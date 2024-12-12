__author__ = "Sanghoon Kang"

import os, sys, getopt
import pandas as pd

def parse_data(in_dir, out_dir):

    # Finding files
    if in_dir.endswith('train'):
        dial_dir = os.path.join(in_dir, 'dialogues_train.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_train.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_train.txt')
        out_path = os.path.join(out_dir, "train.csv")
    elif in_dir.endswith('validation'):
        dial_dir = os.path.join(in_dir, 'dialogues_validation.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_validation.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_validation.txt')
        out_path = os.path.join(out_dir, "validation.csv")
    elif in_dir.endswith('test'):
        dial_dir = os.path.join(in_dir, 'dialogues_test.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_test.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_test.txt')
        out_path = os.path.join(out_dir, "test.csv")
    else:
        print("Cannot find directory")
        sys.exit()

    # Open files
    in_dial = open(dial_dir, 'rt', encoding='utf-8')
    in_emo = open(emo_dir, 'rt', encoding='utf-8')
    in_act = open(act_dir, 'rt', encoding='utf-8')
    
    global_seq = [] 
    local_seq = [] 
    global_emo = [] 
    local_emo = []

    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        seqs = line_dial.split('__eou__')
        seqs = seqs[:-1]

        emos = line_emo.split(' ')
        emos = emos[:-1]

        acts = line_act.split(' ')
        acts = acts[:-1]

        seqs = [s.replace(' .', '.') for s in seqs]
        seqs = [s.replace(' ?', '?') for s in seqs]
        seqs = [s.replace(' !', '!') for s in seqs]
        seqs = [s.replace(' ,', ',') for s in seqs]
        seqs = [s.replace(" ’", "’") for s in seqs]

        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)

        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & action! ", line_count+1, seq_len, emo_len, act_len)
            sys.exit()

        if seq_len < 3: continue # 길이가 3이상인 발화에 대하여

        for index in range(1, seq_len - 1): # index seq 
            global_index_seq = "".join(seqs[index -1 : index + 2]) # 세 개의 발화에 대하여   <sep> token          
            global_index_emo = emos[index]

            for index_ in [index - 1, index + 1]: # pre (index - 1) post (index + 1)
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
    df.to_csv(out_path, encoding = 'utf-8')

def main(argv):

    in_dir = ''
    out_dir = ''

    try:
        opts, args = getopt.getopt(argv,"h:i:o:",["in_dir=","out_dir="])
    except getopt.GetoptError:
        print("python3 parser.py -i <in_dir> -o <out_dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser.py -i <in_dir> -o <out_dir>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg
        elif opt in ("-o", "--out_dir"):
            out_dir = arg

    print("Input directory : ", in_dir)
    print("Output directory: ", out_dir)

    parse_data(in_dir, out_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
    # python daily_parse_v2.py -i train\train -o v2


