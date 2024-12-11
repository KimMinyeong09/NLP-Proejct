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
        
        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)

        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & action! ", line_count+1, seq_len, emo_len, act_len)
            sys.exit()

        # Get rid of the blanks at the start & end of each turns
        seqs = [s.strip() for s in seqs]
        seqs = [s.replace(' .', '.') for s in seqs]
        seqs = [s.replace(' ?', '?') for s in seqs]
        seqs = [s.replace(' !', '!') for s in seqs]
        seqs = [s.replace(' ,', ',') for s in seqs]
        seqs = [s.replace(" ’", "’") for s in seqs]

        for index in range(seq_len): 
            # local
            local_seq.append(seqs[index])
            local_emo.append(emos[index])

    data_dic = {
        'local_seq': local_seq, 
        'local_emo': local_emo
        }
    
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
    # python daily_parse_val.py -i test\test -o parsed_data_val


