import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import random

# ---------------------------
# Dataset and Collation
# ---------------------------
class AmazonDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_input_length=2048, max_target_length=50):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        user_history = sample["user_history"]  # list of item texts
        target_item = sample["target_item"]      # target item text
        
        # Build prompt for user history
        prompt = "This is the summary of a user's purchase history."
        for i, item in enumerate(user_history):
            if i == 0:
                prompt += " The first item bought is as follows. " + item
            else:
                prompt += "\nThe next item bought is as follows. " + item
        # Define a separator before the target item prompt
        sep_token = "\nThe next item bought is as follows.\n"
        # Concatenate the target item (the expected completion)
        full_prompt = prompt + sep_token + target_item
        # print(full_prompt)

        # Tokenize full prompt
        full_encoding = self.tokenizer(full_prompt, truncation=True, max_length=self.max_input_length, return_tensors="pt")
        full_input_ids = full_encoding.input_ids.squeeze(0)  # [seq_len]
        full_attention_mask = full_encoding.attention_mask.squeeze(0)

        # To compute loss only on target tokens, find where the separator ends.
        sep_encoding = self.tokenizer(sep_token, add_special_tokens=False, return_tensors="pt")
        sep_ids = sep_encoding.input_ids.squeeze(0)
        sep_index = self.find_subsequence(full_input_ids, sep_ids)
        # print(sep_index)
        if sep_index is None:
            sep_index = len(full_input_ids) // 2  # fallback if not found

        # Create labels: mask out user history tokens (set to -100)
        labels = full_input_ids.clone()
        labels[:sep_index] = -100

        # Also tokenize the target item text alone (for contrastive branch)
        target_encoding = self.tokenizer(target_item, truncation=True, max_length=self.max_target_length, return_tensors="pt")
        target_input_ids = target_encoding.input_ids.squeeze(0)
        target_attention_mask = target_encoding.attention_mask.squeeze(0)
        
        return {
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
            "labels": labels,
            "sep_index": sep_index,  # integer index marking start of target tokens
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask,
            "user_history": user_history,
            "target_item": target_item
        }
    
    def find_subsequence(self, sequence, subsequence):
        # Look for the first occurrence of subsequence in sequence.
        seq = sequence.tolist()
        sub = subsequence.tolist()
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub:
                # Return the index immediately after the separator (i.e. start of target tokens)
                return i + len(sub)
        return None

def collate_fn(batch):
    # Pad variable-length tensors for a batch.
    full_input_ids = [b["full_input_ids"] for b in batch]
    full_attention_masks = [b["full_attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    sep_indices = [b["sep_index"] for b in batch]
    target_input_ids = [b["target_input_ids"] for b in batch]
    target_attention_masks = [b["target_attention_mask"] for b in batch]

    full_input_ids = nn.utils.rnn.pad_sequence(full_input_ids, batch_first=True, padding_value=0)
    full_attention_masks = nn.utils.rnn.pad_sequence(full_attention_masks, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    target_input_ids = nn.utils.rnn.pad_sequence(target_input_ids, batch_first=True, padding_value=0)
    target_attention_masks = nn.utils.rnn.pad_sequence(target_attention_masks, batch_first=True, padding_value=0)
    
    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": full_attention_masks,
        "labels": labels,
        "sep_indices": torch.tensor(sep_indices, dtype=torch.long),
        "target_input_ids": target_input_ids,
        "target_attention_mask": target_attention_masks
    }

# ---------------------------
# Model Definition
# ---------------------------
class CALRecModel(nn.Module):
    def __init__(self, model_name, projection_dim=128):
        super(CALRecModel, self).__init__()
        self.llama = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.llama.config.hidden_size
        self.user_proj = nn.Linear(hidden_size, projection_dim)
        self.item_proj = nn.Linear(hidden_size, projection_dim)
    
    def forward(self, full_input_ids, full_attention_mask, sep_indices, target_input_ids, target_attention_mask):
        # Forward pass on the concatenated input (user history + target prompt)
        outputs_full = self.llama(input_ids=full_input_ids, attention_mask=full_attention_mask, output_hidden_states=True)
        logits_full = outputs_full.logits            # [batch, seq_len, vocab_size]
        hidden_full = outputs_full.hidden_states[-1]   # [batch, seq_len, hidden_size]
        
        batch_size = full_input_ids.size(0)
        v_U_list = []
        v_T_given_U_list = []
        for i in range(batch_size):
            sep_idx = sep_indices[i].item()
            # Mean-pool the user history tokens (positions before sep_idx)
            user_hidden = hidden_full[i, :sep_idx, :]
            # Mean-pool the target tokens (positions from sep_idx onward)
            target_given_u_hidden = hidden_full[i, sep_idx:, :]
            v_U_list.append(user_hidden.mean(dim=0))
            v_T_given_U_list.append(target_given_u_hidden.mean(dim=0))
        v_U = torch.stack(v_U_list, dim=0)             # [batch, hidden_size]
        v_T_given_U = torch.stack(v_T_given_U_list, dim=0)  # [batch, hidden_size]
        
        # Process the target item text alone
        outputs_target = self.llama(input_ids=target_input_ids, attention_mask=target_attention_mask, output_hidden_states=True)
        hidden_target = outputs_target.hidden_states[-1]  # [batch, target_seq_len, hidden_size]
        v_T = hidden_target.mean(dim=1)                   # [batch, hidden_size]
        
        # Apply projection heads for contrastive losses
        v_U_proj = self.user_proj(v_U)
        v_T_given_U_proj = self.item_proj(v_T_given_U)
        v_T_proj = self.item_proj(v_T)
        
        return logits_full, v_U_proj, v_T_given_U_proj, v_T_proj

# ---------------------------
# Loss Functions
# ---------------------------
def next_item_generation_loss(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def contrastive_loss(emb1, emb2, temperature=0.5):
    emb1_norm = nn.functional.normalize(emb1, dim=-1)
    emb2_norm = nn.functional.normalize(emb2, dim=-1)
    logits = torch.matmul(emb1_norm, emb2_norm.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    return loss

# ---------------------------
# Training Loop
# ---------------------------
def train(model, dataloader, optimizer, device, alpha, beta, temperature, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            full_input_ids = batch["full_input_ids"].to(device)
            full_attention_mask = batch["full_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sep_indices = batch["sep_indices"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            
            optimizer.zero_grad()
            logits_full, v_U, v_T_given_U, v_T = model(full_input_ids, full_attention_mask, sep_indices, target_input_ids, target_attention_mask)
            
            loss_nig = next_item_generation_loss(logits_full, labels)
            loss_tt = contrastive_loss(v_T_given_U, v_T, temperature)
            loss_ut = contrastive_loss(v_U, v_T, temperature)
            
            loss = (1 - alpha - beta) * loss_nig + alpha * loss_tt + beta * loss_ut
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(loss_nig.item(),loss_tt.item(),loss_ut.item(),loss.item())
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")

# ---------------------------
# BM25 Retrieval for Inference
# ---------------------------
def bm25_retrieval(generated_texts, corpus, top_n=10, epsilon=1/5000):
    tokenized_corpus = [doc.split() if isinstance(doc, str) else doc for doc in corpus]
    # print(tokenized_corpus[:5])
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    candidate_scores = np.zeros(len(corpus))
    
    for text, score in generated_texts:
        tokenized_query = text.split() if isinstance(text, str) else text
        # print(tokenized_query)

        bm25_scores = bm25.get_scores(tokenized_query)
        # Scale BM25 scores to [0, 1]
        bm25_scores_scaled = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
        )
        modulated_scores = np.exp(epsilon * score) * bm25_scores_scaled
        candidate_scores = np.maximum(candidate_scores, modulated_scores)
    
    # Get indices of the top_n candidates.
    top_indices = np.argsort(candidate_scores)[::-1][:top_n]
    return top_indices, candidate_scores[top_indices]


# ---------------------------
# Evaluation Functions
# ---------------------------
def evaluate(model, dataset, corpus, tokenizer, device, num_return_sequences=32, num_preds=10, epsilon=1/5000):
    model.eval()
    total = 0
    recall_at_1 = 0.0
    recall_at_10 = 0.0
    ndcg_at_10 = 0.0
    mrr = 0.0
    
    for sample in tqdm(dataset, desc="Evaluating"):
        # Construct prompt from user history (exclude target)
        user_history = sample["user_history"]
        prompt = "This is the summary of a user's purchase history."
        for i, item in enumerate(user_history):
            if i == 0:
                prompt += " The first item bought is as follows. " + item
            else:
                prompt += " The next item bought is as follows. " + item
        
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = encoding.input_ids  # Save for later token slicing
        generated_outputs = model.llama.generate(
            **encoding, 
            max_length=input_ids.shape[1] + 50,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            temperature=0.5,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        generated_texts = []
        seqs   = generated_outputs.sequences.cpu()           # (B * R, L_out)
        scores = generated_outputs.sequences_scores.cpu()    # (B * R,)

        for i, output in enumerate(seqs):
            # Remove the prompt tokens by slicing from the input length onward
            answer_tokens = output[input_ids.shape[1]:]
            text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            # For demonstration, we use a dummy log-probability.
            log_prob = scores[i].item()
            generated_texts.append((text, log_prob))
        
        # Retrieve top candidate items using BM25 retrieval
        top_indices, _ = bm25_retrieval(generated_texts, corpus, top_n=num_preds, epsilon=epsilon)
        ranked_candidates = [corpus[idx] for idx in top_indices]
        
        # Compare against ground truth target
        gt = sample["target_item"].strip()
        rank = None
        for idx, candidate in enumerate(ranked_candidates, start=1):
            # print(candidate)
            if candidate.strip() == gt:
                rank = idx
                break
        if rank is None:
            rank = num_preds + 1  # not found in top candidates
        
        # Update Recall
        if rank <= 1:
            recall_at_1 += 1
        if rank <= 10:
            recall_at_10 += 1
        
        # Update MRR
        mrr += 1.0 / rank
        
        # Update NDCG@10: if ground truth is found in top 10, DCG = 1/log2(rank+1) (IDCG = 1)
        if rank <= 10:
            ndcg_at_10 += 1.0 / np.log2(rank + 1)
        
        total += 1
    
    recall_at_1 /= total
    recall_at_10 /= total
    mrr /= total
    ndcg_at_10 /= total
    
    print(f"Evaluation Metrics on {total} samples:")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print(f"NDCG@10: {ndcg_at_10:.4f}")
    print(f"MRR: {mrr:.4f}")

class MultiCategoryDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (List[Dataset]): A list of dataset instances.
        """
        self.datasets = datasets
        self.idx_map = []  # Maps global idx to (dataset_idx, sample_idx)

        # Build flat index mapping
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.idx_map.append((dataset_idx, sample_idx))

        # Shuffle index map initially
        random.shuffle(self.idx_map)

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.idx_map[idx]
        return self.datasets[dataset_idx][sample_idx]

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/amazon_data.jsonl", help="Path to the training data (JSONL format)")
    parser.add_argument("--test_data_file", type=str, default="data/amazon_test.jsonl", help="Path to the test data (JSONL format)")
    parser.add_argument("--corpus_file", type=str, default="data/amazon_item_corpus.txt", help="Path to the item corpus file (one item per line)")
    parser.add_argument("--model_name", type=str, default="/models/smollm", help="Pretrained LLAMA model name or path")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.125, help="Weight for contrastive loss L_TT")
    parser.add_argument("--beta", type=float, default=-0.025, help="Weight for contrastive loss L_UT")
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = CALRecModel(args.model_name)
    for param in model.llama.parameters():
        param.requires_grad = False
    for param in model.llama.lm_head.parameters():
        param.requires_grad = True
    # model.load_state_dict(torch.load("calrec_finetuned.pt", weights_only=True))
    model.to(device)
    
    
    # # Training
    datasets = []
    dataset_names = ["Subscription_Boxes", "Magazine_Subscriptions"]
    for dataset_name in dataset_names:
        dataset_path = f"{dataset_name}.jsonl"
        if os.path.exists(dataset_path):
            datasets.append(AmazonDataset(
                data_file=dataset_path,
                tokenizer=tokenizer,
            ))
        else:
            print(f"Dataset file {dataset_path} not found. Skipping this dataset.") 
    train_dataset = MultiCategoryDataset(datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.learning_rate)
    train(model, train_dataloader, optimizer, device, args.alpha, args.beta, args.temperature, args.num_epochs)
    
    # Save the finetuned model
    model_save_path = "models/calrec_finetuned.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Load item corpus for retrieval evaluation
    if os.path.exists(args.corpus_file):
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f.readlines()]
    else:
        print("Corpus file not found. Exiting evaluation.")
        return
    
    # Evaluation
    if os.path.exists(args.test_data_file):
        test_dataset = AmazonDataset(args.test_data_file, tokenizer)
        # For evaluation, we iterate sample by sample.
        test_data = [test_dataset[i] for i in range(len(test_dataset))]
        # test_data = [test_dataset[i] for i in range(40,100)]
        evaluate(model, test_data, corpus, tokenizer, device)
    else:
        print("Test data file not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
