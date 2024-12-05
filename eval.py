import torch 

# Validation function
def evaluate_model(model, validation_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in validation_loader:
            global_input_ids = batch["global_input_ids"].to(device)
            global_attention_mask = batch["global_attention_mask"].to(device)
            global_emo = batch["global_emo"].to(device)

            local_input_ids = batch["local_input_ids"].to(device)
            local_attention_mask = batch["local_attention_mask"].to(device)
            local_emo = batch["local_emo"].to(device)

            # Forward pass
            global_logits = model(
                input_ids=global_input_ids, 
                attention_mask=global_attention_mask
            ).logits

            local_logits = model(
                input_ids=local_input_ids, 
                attention_mask=local_attention_mask
            ).logits

            # Classification loss
            loss = torch.nn.CrossEntropyLoss()
            global_loss = loss(global_logits, global_emo)
            local_loss = loss(local_logits, local_emo)
            total_loss += global_loss.item() + local_loss.item()

            # Accuracy calculation
            global_preds = torch.argmax(global_logits, dim=-1)
            local_preds = torch.argmax(local_logits, dim=-1)

            total_correct += (global_preds == global_emo).sum().item()
            total_correct += (local_preds == local_emo).sum().item()
            total_samples += 2 * global_emo.size(0)  # (global + local)

    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy