from sklearn.metrics import f1_score, precision_score, recall_score
import json 
import torch 
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from dataset import preprocess_function, EmotionDataset
from datasets import load_dataset


DNAME = 'Daily' ## Daily or EMO
NUM_CLASSES = 7 
LABELS_DAILY = {
    0: 'no emotion',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'sadness',
    6: 'surprise'
}

LABELS_EMO = {
    0: 'neutral',
    1: 'fearful',
    2: 'dissatisfied',
    3: 'apologetic',
    4: 'abusive', 
    5: 'excited',
    6: 'satisfied', 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "" ## 저장된 checkpoint path 
model = RobertaForSequenceClassification.from_pretrained(checkpoint_path, num_labels=NUM_CLASSES)
model.to(device)


## dataset 
test_dataset = load_dataset("csv", data_files="parsed_data/test.csv")['train']
test_dataset = test_dataset.remove_columns('Unnamed: 0')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
test_dataset = test_dataset.map(
    preprocess_function, 
    batched=True, 
    fn_kwargs={
        "tokenizer": tokenizer  # 미리 정의된 tokenizer 전달
    })
test_dataset = EmotionDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Validation function
outputs = [] 

for batch in test_loader:
    global_input_ids = batch["global_input_ids"].to(device)
    global_attention_mask = batch["global_attention_mask"].to(device)
    global_emo = batch["global_emo"].to(device)

    local_input_ids = batch["local_input_ids"].to(device)
    local_attention_mask = batch["local_attention_mask"].to(device)
    local_emo = batch["local_emo"].to(device)

    with torch.no_grad():
        # Forward pass
        local_logits = model(
            input_ids=local_input_ids,
            attention_mask=local_attention_mask
        ).logits

        # Accuracy calculation
        local_preds = torch.argmax(local_logits, dim=-1)

        for idx in range(len(local_emo)): 
            outputs.append({'predictions': local_preds[idx], 'labels': local_emo[idx]})


# evaluate 
predictions = torch.cat([o['predictions'] for o in outputs], dim = 0)
labels = torch.cat([o['labels'] for o in outputs], dim = 0)

y_hat_predictions = predictions.cpu().numpy()
y_labels = torch.argmax(labels, dim = 1).cpu().numpy()

metrics = {
    "macro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average = 'macro', zero_division = 0))
}


# metrics for label
if DNAME == 'Daily':
    LABELS_DIC = LABELS_DAILY
elif DNAME == 'EMO':
    LABELS_DIC = LABELS_EMO

for i, label in LABELS_DIC.items():
    metrics[label+"-precision"] = precision_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
    metrics[label+"-recall"] = recall_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
    metrics[label+"-f1"] = f1_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]

with open(f'eval_{DNAME}.json', 'w') as f: 
    json.dump(metrics, f, indent = 4)
