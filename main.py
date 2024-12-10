from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch 
import torch.optim as optim
from torch.nn import functional as F

from dataset import preprocess_function, EmotionDataset
from eval import evaluate_model

DNAME = 'Daily' ## Daily or Emo

train_dataset = load_dataset("csv", data_files=f"dataset/parsed_data/{DNAME}/train.csv")['train']
valid_dataset = load_dataset("csv", data_files=f"dataset/parsed_data/{DNAME}/validation.csv")['train']

train_dataset = train_dataset.remove_columns('Unnamed: 0')
valid_dataset = valid_dataset.remove_columns('Unnamed: 0')

print(train_dataset)
print(valid_dataset)

# 로버타 토크나이저 불러오기
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 데이터셋 전처리 적용
train_dataset = train_dataset.map(
    preprocess_function, 
    batched=True, 
    fn_kwargs={
        "tokenizer": tokenizer  # 미리 정의된 tokenizer 전달
    })
valid_dataset = valid_dataset.map(    
    preprocess_function, 
    batched=True, 
    fn_kwargs={
        "tokenizer": tokenizer  # 미리 정의된 tokenizer 전달
    })

# PyTorch 데이터셋 생성
train_dataset = EmotionDataset(train_dataset)
valid_dataset = EmotionDataset(valid_dataset)

## parameter 
BATCH_SIZE = 4
EPOCHS = 50
LR = 5e-5
NUM_CLASSES = 7 # 감정 레이블 개수 
ema_decay = 0.99 

# 모델 셋업
teacher_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_CLASSES)
student_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# optimizer, scheduler 생성 
optimizer = optim.Adam(student_model.parameters(), lr = LR)
scheduler = optim.lr_scheduler.StepLR(
    optimizer = optimizer, 
    step_size = 10, 
    gamma = 0.5
)

## train 
for epoch in range(EPOCHS): 
    student_model.train()
    teacher_model.eval() # it is not for train 
    epoch_loss = 0.0 

    for idx, batch in enumerate(train_loader): 
        global_input_ids = batch['global_input_ids'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        global_emo = batch['global_emo'].to(device)

        local_input_ids = batch['local_input_ids'].to(device)
        local_attention_mask = batch['local_attention_mask'].to(device)
        local_emo = batch['local_emo'].to(device)

        pred_global_logits = teacher_model(
            input_ids = global_input_ids, 
            attention_mask = global_attention_mask, 
        ).logits

        pred_local_logits = student_model(
            input_ids = local_input_ids, 
            attention_mask = local_attention_mask
        ).logits

        # loss 
        loss = torch.nn.CrossEntropyLoss()

        loss_global = loss(pred_global_logits, global_emo)

        loss_local = loss(pred_local_logits, local_emo)

        total_loss = loss_global + loss_local

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # EMA 업데이트 
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha = 1 - ema_decay)

        epoch_loss += total_loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    val_loss, val_accuracy = evaluate_model(student_model, valid_loader, device)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss : {avg_train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    student_model.save_pretrained(f"{DNAME}_model_{epoch+1}_epoch")