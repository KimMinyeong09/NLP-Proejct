import torch 

def preprocess_function(examples, tokenizer = None):
    # 글로벌 입력 시퀀스 처리
    global_encodings = tokenizer(
        examples['global_seq'],
        truncation=True,
        padding='max_length',
        max_length=512,
        add_special_tokens=False # 데이터셋에 special token 있음. 
    )
    # 로컬 입력 시퀀스 처리
    local_encodings = tokenizer(
        examples['local_seq'],
        truncation=True,
        padding='max_length',
        max_length=512,
        add_special_tokens=True # 데이터셋에 special token 없음. 자동으로 토큰 추가 
    )

    return {
        'global_input_ids': global_encodings['input_ids'],
        'global_attention_mask': global_encodings['attention_mask'],
        'global_emo': examples['global_emo'],  # 글로벌 레이블 
        'local_input_ids': local_encodings['input_ids'],
        'local_attention_mask': local_encodings['attention_mask'],
        'local_emo': examples['local_emo'],    # 로컬 레이블 
    }

# PyTorch 데이터셋 변환
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, flag_test = False):
        self.dataset = dataset
        self.flag_test = flag_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.flag_test: 
            item = {
            'local_input_ids': torch.tensor(self.dataset[idx]['local_input_ids']),
            'local_attention_mask': torch.tensor(self.dataset[idx]['local_attention_mask']),
            'local_emo': torch.tensor(self.dataset[idx]['local_emo']),
            }
        else:
            item = {
                'global_input_ids': torch.tensor(self.dataset[idx]['global_input_ids']),
                'global_attention_mask': torch.tensor(self.dataset[idx]['global_attention_mask']),
                'global_emo': torch.tensor(self.dataset[idx]['global_emo']),
                'local_input_ids': torch.tensor(self.dataset[idx]['local_input_ids']),
                'local_attention_mask': torch.tensor(self.dataset[idx]['local_attention_mask']),
                'local_emo': torch.tensor(self.dataset[idx]['local_emo']),
            }
        return item