import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, TrainerState, TrainerControl
from accelerate import Accelerator
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm.auto import tqdm
import bm25s
from bm25s.tokenization import Tokenizer
import argparse
import json
import numpy as np
from trl import SFTConfig, SFTTrainer
from PIL import Image

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

# ---------------------------
# Dataset and Collation
# ---------------------------
from collections import defaultdict

class AmazonDataset(Dataset):
    def __init__(self, data_file, processor, max_input_length=512, max_target_length=50):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        user_history = sample["user_history"]  # list of item texts
        target_item = sample["target_item"]      # target item text
        user_history_images = sample.get("user_history_images", [])  # optional list of image paths
        target_item_image = sample.get("target_item_image", "")  # optional target image path

        length = torch.randint(1, min(6, len(user_history) + 1), (1,)).item()
        start_idx = torch.randint(0, len(user_history) - length + 1, (1,)).item()

        history_window = user_history + [target_item]
        images_window = user_history_images + [target_item_image]
        history_window = history_window[start_idx:start_idx + length + 1]
        images_window = images_window[start_idx:start_idx + length + 1]

        user_history = history_window[:-1]
        target_item = history_window[-1]
        user_history_images = images_window[:-1]
        target_item_image = images_window[-1]
        
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
        target_prompt = target_item

        # print(len(user_history_images))
        hist_imgs = []
        for img_path in user_history_images:
            img = Image.open(img_path).convert("RGB")
            hist_imgs.append(img)
        # print(len(hist_imgs))

        # target image
        tgt_img = Image.open(target_item_image).convert("RGB")

        # full images
        full_imgs = hist_imgs + [tgt_img]

        example = {
            # "input_prompt": prompt + sep_token,
            # "input_images": hist_imgs,
            "full_prompt": full_prompt,
            "target_prompt": target_prompt,
            "full_images": full_imgs,   # list of (C,H,W)
            "target_image": tgt_img,       # (C,H,W)
        }
        return example
    
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

def format_input(sample):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": sample["input_prompt"],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["target_prompt"]}],
        },
    ]

def format_target(sample):
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["target_prompt"],
                }
            ],
        },
    ]

def collate_fn(batch, processor):
    """
    Pads variable‑length text tensors, stacks the *last* target image,
    and leaves history_images as a list of lists (your model can decide
    whether to average them or process sequentially).
    """
    full_imgs = [b["full_images"] for b in batch]
    full_prompt = [b["full_prompt"] for b in batch]
    target_img = [b["target_image"] for b in batch]
    target_prompt = [b["target_prompt"] for b in batch]

    # process full prompt
    # print(len(full_imgs), len(full_imgs[0]), np.array(full_imgs[0][0]).shape)
    # full_prompt = [processor.apply_chat_template(format_input(b), tokenize=False, add_generation_prompt=True) for b in batch]
    # target_prompt = [processor.apply_chat_template(format_target(b), tokenize=False, add_generation_prompt=True) for b in batch]
    # print(full_prompt[0])

    full_enc = processor(
        images=full_imgs,
        text=full_prompt,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        # max_length=self.max_input_length
    )

    full_input_ids       = full_enc.input_ids
    full_attention_mask  = full_enc.attention_mask
    labels = full_input_ids.clone()

    tokenizer = processor.tokenizer
    sep_token = "\nThe next item bought is as follows.\n"
    # locate sep_index to mask history in labels
    sep_enc = tokenizer(sep_token, add_special_tokens=False, return_tensors="pt")
    sep_ids = sep_enc.input_ids[0].tolist()
    
    sep_indices = []
    batch_size = full_input_ids.size(0)
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    # find where sep_ids appears in full_input_ids
    for k in range(batch_size):
        seq_ids = full_input_ids[k].tolist()
        sep_index = None
        for i in range(len(seq_ids) - len(sep_ids) + 1):
            if seq_ids[i : i + len(sep_ids)] == sep_ids:
                sep_index = i + len(sep_ids)
                break
        if sep_index is None:
            sep_index = len(full_input_ids) // 2

        sep_indices.append(sep_index)

        # labels: -100 for history tokens
        labels[k][:sep_index] = -100
        labels[k][labels[k] == processor.tokenizer.pad_token_id] = -100
        labels[k][labels[k] == image_token_id] = -100

    # process target alone (for contrastive branch)
    tgt_enc = processor(
        images=target_img,
        text=target_prompt,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        # max_length=self.max_target_length
    )
    target_input_ids      = tgt_enc.input_ids
    target_attention_mask = tgt_enc.attention_mask

    full_input_ids       = pad_sequence(full_input_ids, batch_first=True, padding_value=-100)
    full_attention_mask  = pad_sequence(full_attention_mask, batch_first=True, padding_value=0)
    labels               = pad_sequence(labels, batch_first=True, padding_value=-100)
    target_input_ids     = pad_sequence(target_input_ids, batch_first=True, padding_value=-100)
    target_attention_mask = pad_sequence(target_attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "labels": labels,
        "sep_indices": torch.LongTensor(sep_indices),
        "target_input_ids": target_input_ids,
        "target_attention_mask": target_attention_mask,
        "target_images": tgt_enc.pixel_values,
        "images": full_enc.pixel_values,
        # "input_text": [b["input_text"] for b in batch],
        "target_text": target_prompt
    }

# ---------------------------
# Model Definition
# ---------------------------

class PixRecModel(nn.Module):
    def __init__(
        self,
        model_path: str = "models/smolvlm256",
        projection_dim: int = 128,
        is_peft: bool = False,
        model_save_path = None
    ):
        super().__init__()
        # Load multimodal causal LM (PaLI-Gemma)
        from transformers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_path,
            # torch_dtype=torch.float16,
            # _attn_implementation="flash_attention_2",
            quantization_config=nf4_config,
            device_map="auto"
        )
        self.config = self.llm.config
        if is_peft:
            # Identify linear layers for LoRA adapters
            def find_all_linear_names(model):
                cls = nn.Linear
                names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        parts = name.split('.')
                        names.add(parts[-1])
                # names.discard('lm_head')
                names.discard('proj')
                names.discard('o_proj')
                names.discard('v_proj')
                names.discard('q_proj')
                names.discard('k_proj')
                print(list(names))
                return list(names)

            target_modules = find_all_linear_names(self.llm)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                # use_dora=True,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            if model_save_path:
                self.llm = PeftModel.from_pretrained(self.llm, os.path.join(model_save_path, "peft"))
            else:
                self.llm = prepare_model_for_kbit_training(self.llm)
                self.llm = get_peft_model(self.llm, peft_config)

        # self.hidden_size = self.llm.config.hidden_size # For Paligemma
        self.hidden_size = self.llm.config.text_config.hidden_size # For SmolVLM
        # print(self.hidden_size)
        self.user_proj = nn.Linear(self.hidden_size, projection_dim)
        self.item_proj = nn.Linear(self.hidden_size, projection_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        sep_indices: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_images: torch.Tensor = None
    ):
        # Default target_images to full_images if not provided
        if target_images is None:
            target_images = images

        # 1) Multimodal forward pass on (history + target) concatenation
        out_full = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            output_hidden_states=True,
        )
        logits_full = out_full.logits                     # (B, seq_len, vocab_size)
        hidden_full = out_full.hidden_states[-1]         # (B, seq_len, hidden_size)
        hidden_full = hidden_full[:,:,:self.hidden_size]  
        # print(hidden_full.shape)
        
        # Split and pool: user history vs. target tokens
        batch_size = input_ids.size(0)
        v_U_list, v_T_given_U_list = [], []
        for i in range(batch_size):
            sep_idx = sep_indices[i].item()
            user_hidden = hidden_full[i, :sep_idx, :]
            targ_hidden = hidden_full[i, sep_idx:, :]
            v_U_list.append(user_hidden.mean(dim=0))
            v_T_given_U_list.append(targ_hidden.mean(dim=0))

        v_U = torch.stack(v_U_list, dim=0)               # (B, hidden_size)
        v_T_given_U = torch.stack(v_T_given_U_list, dim=0)  # (B, hidden_size)
        # print(v_U.shape, v_T_given_U.shape)
        
        # 2) Multimodal forward on target prompt + image alone
        out_tgt = self.llm(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            pixel_values=target_images,
            output_hidden_states=True,
        )
        hidden_tgt = out_tgt.hidden_states[-1]           # (B, tgt_seq_len, hidden_size)
        hidden_tgt = hidden_tgt[:,:,:self.hidden_size]
        v_T = hidden_tgt.mean(dim=1)                     # (B, hidden_size)
        # print(v_T.shape,hidden_tgt.shape)
        
        # Projection for contrastive objectives
        v_U_proj = self.user_proj(v_U)                   # (B, projection_dim)
        v_T_given_U_proj = self.item_proj(v_T_given_U)
        v_T_proj = self.item_proj(v_T)

        return logits_full, v_U_proj, v_T_given_U_proj, v_T_proj

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.config.use_cache = False   # HF requirement
        return self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.llm.gradient_checkpointing_disable()
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

accelerator = Accelerator()

def train(
    model,
    dataloader,
    optimizer,
    alpha: float,
    beta: float,
    temperature: float,
    num_epochs: int,
    model_save_path = None,
    device = None
):

    # Prepare everything
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_nig,epoch_loss_tt, epoch_loss_ut = 0.0, 0.0, 0.0

        # tqdm will only show on the main process
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            full_input_ids        = batch["input_ids"]
            full_attention_mask   = batch["attention_mask"]
            full_images           = batch["images"]
            sep_indices           = batch["sep_indices"]
            target_input_ids      = batch["target_input_ids"]
            target_attention_mask = batch["target_attention_mask"]
            target_images         = batch["target_images"]
            labels                = batch["labels"]

            # forward
            # logits_full, v_U, v_T_given_U, v_T = model(
            #     full_input_ids.to(device),
            #     full_attention_mask.to(device),
            #     full_images.to(device),
            #     sep_indices.to(device),
            #     target_input_ids.to(device),
            #     target_attention_mask.to(device),
            #     target_images.to(device),
            # )
            
            logits_full, v_U, v_T_given_U, v_T = model(
                full_input_ids,
                full_attention_mask,
                full_images,
                sep_indices,
                target_input_ids,
                target_attention_mask,
                target_images,
            )

            # losses
            # print(logits_full, labels)
            loss_nig = next_item_generation_loss(logits_full, labels)
            loss_tt  = contrastive_loss(v_T_given_U, v_T, temperature)
            loss_ut  = contrastive_loss(v_U, v_T, temperature)
            loss     = (1 - alpha - beta) * loss_nig + alpha * loss_tt + beta * loss_ut

            # backward + step
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_loss_nig += loss_nig.item()
            epoch_loss_tt += loss_tt.item()
            epoch_loss_ut += loss_ut.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_loss_nig = epoch_loss_nig / len(dataloader)
        avg_loss_tt = epoch_loss_tt / len(dataloader)
        avg_loss_ut = epoch_loss_ut / len(dataloader)
        # print only on the main process
        accelerator.print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f},avg loss NIG: {avg_loss_nig:.4f}, avg loss TT: {avg_loss_tt:.4f}, avg loss UT: {avg_loss_ut:.4f}")
        # torch.save(model.state_dict(), model_save_path)
        accelerator.save_model(model, model_save_path)


def peft_train(model, processor, train_dataset, test_dataset, alpha, beta, temperature, num_epochs, batch_size, model_save_path):

    # Setting Hyperparamter
    # training_arguments = TrainingArguments(
    #     output_dir=model_save_path,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=1,
    #     gradient_accumulation_steps=1,
    #     optim="adamw_torch",
    #     num_train_epochs=num_epochs,
    #     do_eval=False,
    #     eval_strategy="no",
    #     # eval_steps=0.2,
    #     # logging_steps=1000,
    #     logging_strategy="epoch",
    #     save_strategy="epoch",
    #     warmup_steps=10,
    #     learning_rate=1e-4,
    #     group_by_length=False,
    #     report_to="none",
    #     remove_unused_columns=False,
    #     dataloader_pin_memory=True,
    # )

    training_arguments = SFTConfig(
        output_dir=model_save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch_fused",
        report_to="none",
        # fp16=True,
        # bf16=True,
        remove_unused_columns=False,
        average_tokens_across_devices=False,
        gradient_checkpointing=True,
        # dataset_text_field="",
        group_by_length=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=True,
        save_safetensors=False,
        # use_cpu=True
    )

    class CustomTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            full_input_ids = inputs["input_ids"]
            full_attention_mask = inputs["attention_mask"]
            full_images = inputs["images"]
            labels = inputs["labels"]
            sep_indices = inputs["sep_indices"]
            target_input_ids = inputs["target_input_ids"]
            target_attention_mask = inputs["target_attention_mask"]
            target_images = inputs["target_images"]
            outputs = model(
                full_input_ids,
                full_attention_mask,
                full_images,
                sep_indices,
                target_input_ids,
                target_attention_mask,
                target_images,
            )
            logits_full, v_U, v_T_given_U, v_T = outputs
            loss_nig = next_item_generation_loss(logits_full, labels)
            loss_tt = contrastive_loss(v_T_given_U, v_T, temperature)
            loss_ut = contrastive_loss(v_U, v_T, temperature)
            
            loss = (1 - alpha - beta) * loss_nig + alpha * loss_tt + beta * loss_ut

            return (loss, outputs) if return_outputs else loss
    
    class AutoSavePEFTCallback(TrainerCallback):
        def __init__(self, model_getter, peft_model_getter, output_dir: str):
            self.model_getter = model_getter
            self.peft_model_getter = peft_model_getter
            self.output_dir = output_dir

        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Get model + PEFT model
            model = self.model_getter()
            peft_model = self.peft_model_getter()

            checkpoint_dir = self.output_dir

            # Save PEFT adapter
            peft_save_dir = os.path.join(checkpoint_dir, "peft")
            peft_model.save_pretrained(peft_save_dir)
            print(f"[Checkpoint {state.global_step}] Saved PEFT adapter to {peft_save_dir}")

            # Save full model weights
            # full_model_path = os.path.join(checkpoint_dir, "full_model_state.pth")
            # torch.save(model.state_dict(), full_model_path)
            # print(f"[Checkpoint {state.global_step}] Saved full model to {full_model_path}")

            # Save model proj weights
            proj_model_path = os.path.join(checkpoint_dir, "proj_model_state.pth")
            proj_state = {
                'user_proj': model.user_proj.state_dict(),
                'item_proj': model.item_proj.state_dict(),
            }
            torch.save(proj_state, proj_model_path)
            print(f"[Checkpoint {state.global_step}] Saved proj model to {proj_model_path}")

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=lambda x: collate_fn(x, processor),
        callbacks=[
            AutoSavePEFTCallback(
                model_getter=lambda: model,
                peft_model_getter=lambda: model.llm,
                output_dir=training_arguments.output_dir
            )
        ]
    )

    model.llm.config.use_cache = False
    trainer.train()

# ---------------------------
# BM25 Retrieval for Inference
# ---------------------------
def bm25_retrieval(generated_texts, corpus, top_n=10, epsilon=1/5000):
    tokenizer = Tokenizer()
    tokenized_corpus = tokenizer.tokenize(corpus, return_as="tuple")

    generated_texts.sort(key=lambda x: x[1], reverse=True)
    generated_texts = list(set(generated_texts))  # remove duplicates
    
    # bm25 = BM25Okapi(tokenized_corpus)

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokenized_corpus)

    generated_texts = generated_texts[:top_n]
    
    candidate_scores = np.zeros((len(corpus), top_n))
    
    for i in range(top_n):
        text = generated_texts[i][0]
        score = generated_texts[i][1]
        tokenized_query = tokenizer.tokenize([text], return_as="tuple")
        # print(tokenized_query)

        # bm25_scores = bm25.get_scores(tokenized_query)
        bm25_scores = retriever.get_scores(tokenized_query.ids[0])
        # Scale BM25 scores to [0, 1]
        bm25_scores_scaled = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
        )
        modulated_scores = np.exp(epsilon * score) * bm25_scores_scaled
        candidate_scores[:, i] = modulated_scores
    
    # Get indices of the top_n candidates.
    candidate_scores = np.max(candidate_scores, axis=1)
    top_indices = np.argsort(candidate_scores)[::-1]
    return top_indices, candidate_scores[top_indices]

# ---------------------------
# Evaluation Functions
# ---------------------------

def evaluate(
    model,
    dataloader,
    corpus,
    processor,
    device,
    num_return_sequences: int = 32,
    num_preds: int = 20,
    epsilon: float = 1/5000
):
    # model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    total = 0
    recall_at_1 = 0.0
    recall_at_10 = 0.0
    mrr = 0.0
    ndcg_at_10 = 0.0

    for batch in tqdm(dataloader, desc="Evaluating"):
        # # unpack & move to GPU
        full_input_ids        = batch["input_ids"].to(device)      # (B, L_full)
        full_attention_mask   = batch["attention_mask"].to(device) # (B, L_full)
        full_images           = batch["images"].to(device)              # (B, C, H, W)
        sep_indices           = batch["sep_indices"].to(device)         # (B,)
        # target_input_ids      = batch["target_input_ids"].to(device)    # (B, L_tgt)
        # target_attention_mask = batch["target_attention_mask"].to(device)
        # target_images         = batch.get("target_images", full_images).to(device)
        gt_texts              = batch["target_text"]                    # list[str], len B

        # full_input_ids        = batch["input_ids"]      # (B, L_full)
        # full_attention_mask   = batch["attention_mask"] # (B, L_full)
        # full_images           = batch["images"]              # (B, C, H, W)
        # sep_indices           = batch["sep_indices"]        # (B,)
        # # target_input_ids      = batch["target_input_ids"]   # (B, L_tgt)
        # # target_attention_mask = batch["target_attention_mask"]
        # # target_images         = batch.get("target_images", full_images)
        # gt_texts              = batch["target_text"]                    # list[str], len B

        batch_size = full_input_ids.size(0)

        input_ids = []
        input_attention_mask = []
        input_images = []
        # print(full_input_ids.shape, full_attention_mask.shape, full_images.shape)
        for i in range(batch_size):
            sep_idx = sep_indices[i].item()
            input_ids.append(full_input_ids[i, :sep_idx])
            input_attention_mask.append(full_attention_mask[i, :sep_idx])
            # input_images.append(full_images[:-1, :, :, :]) # For Paligemma
            input_images.append(full_images[i,:-1, :, :, :]) # For SmolVLM

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=model.llm.config.pad_token_id, padding_side='left')
        input_attention_mask = torch.nn.utils.rnn.pad_sequence(input_attention_mask, batch_first=True, padding_value=model.llm.config.pad_token_id, padding_side='left')
        # input_ids = torch.stack(input_ids, dim=0)
        # input_attention_mask = torch.stack(input_attention_mask, dim=0)
        input_images = torch.stack(input_images, dim=0) 
        # input_images = input_images.squeeze(0) # For Paligemma
        # print(input_ids.shape, input_attention_mask.shape, input_images.shape)

        # --- generation (multimodal) ---
        gen_out = model.llm.generate(
            input_ids=input_ids.to(device),
            attention_mask=input_attention_mask.to(device),
            pixel_values=input_images.to(device),
            max_new_tokens=50,
            temperature=0.5,
            repetition_penalty=1.2,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        seqs   = gen_out.sequences           # (B * R, L_out)
        scores = gen_out.sequences_scores    # (B * R,)

        # reshape: [B, R, L_out]
        seqs   = seqs.view(batch_size, num_return_sequences, -1)
        scores = scores.view(batch_size, num_return_sequences)

        # --- per‑sample ranking & metrics ---
        for i in range(batch_size):
            gen_texts = []
            prompt_len = input_ids.size(1)
            # print("gt:", gt_texts[i])
            for j in range(num_return_sequences):
                out_ids   = seqs[i, j, prompt_len:]                     # drop prompt tokens
                text      = processor.decode(out_ids, skip_special_tokens=True)
                # print(f"text{j}:", text)
                logp      = scores[i, j].item()
                gen_texts.append((text, logp))

            # BM25 retrieval over your corpus
            top_idxs, _ = bm25_retrieval(gen_texts, corpus,
                                         top_n=num_preds,
                                         epsilon=epsilon)
            ranked = [corpus[idx] for idx in top_idxs]
            gt = gt_texts[i].strip()

            # find rank
            try:
                rank = next(r+1 for r, cand in enumerate(ranked)
                            if cand.strip() == gt)
            except StopIteration:
                rank = num_preds + 1

            # update stats
            if rank <= 1:   recall_at_1 += 1
            if rank <= 10:  recall_at_10 += 1
            mrr       += 1.0 / rank
            if rank <= 10:
                ndcg_at_10 += 1.0 / np.log2(rank + 1)

            total += 1

    # normalize
    recall_at_1  /= total
    recall_at_10 /= total
    mrr          /= total
    ndcg_at_10   /= total

    print(f"Evaluated {total} samples")
    print(f"Recall@1:  {recall_at_1:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print(f"MRR:       {mrr:.4f}")
    print(f"NDCG@10:   {ndcg_at_10:.4f}")

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_file", type=str, default="amazon_data.jsonl", help="Path to the training data (JSONL format)")
    # parser.add_argument("--test_data_file", type=str, default="amazon_test.jsonl", help="Path to the test data (JSONL format)")
    # parser.add_argument("--corpus_file", type=str, default="amazon_item_corpus.txt", help="Path to the item corpus file (one item per line)")
    parser.add_argument("--model_name", type=str, default="models/paligemma2", help="Pretrained LLM model name or path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.125, help="Weight for contrastive loss L_TT")
    parser.add_argument("--beta", type=float, default=-0.025, help="Weight for contrastive loss L_UT")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="train", help="Modes can be train or test")
    parser.add_argument("--peft", type=bool, default=False, help="Normal or PEFT training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
    )
    processor.image_processor.do_image_splitting = False

    model = PixRecModel(args.model_name, is_peft = args.peft)
    model = model.to(device)
    print(
        f"Memory footprint: {model.llm.get_memory_footprint() / 1024 **3:.2f} GB."
    )

    model_save_path = "./models/amazmodel_peft/" if args.peft else "./models/amazmodel/"

    datasets = []
    dataset_names = ["Subscription_Boxes", "Magazine_Subscriptions"]
    for dataset_name in dataset_names:
        dataset_path = f"data/{dataset_name}.jsonl"
        if os.path.exists(dataset_path):
            datasets.append(AmazonDataset(
                data_file=dataset_path,
                processor=processor,
            ))
        else:
            print(f"Dataset file {dataset_path} not found. Skipping this dataset.") 

    train_dataset = MultiCategoryDataset(datasets)
    
    if args.mode == "train":            
        if not args.peft:
            for param in model.llm.parameters():
                param.requires_grad = False
            for param in model.llm.lm_head.parameters():
                param.requires_grad = True
            print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 
            # Training from Scratch
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, processor))
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
            train(model, train_dataloader, optimizer, args.alpha, args.beta, args.temperature, args.num_epochs, model_save_path, device)
            accelerator.save_model(model, model_save_path)
            print(f"Model saved to {model_save_path}")

        else:
            print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 
            # PEFT training
            peft_train(model, processor, train_dataset, train_dataset, args.alpha, args.beta, args.temperature, args.num_epochs, args.batch_size, model_save_path)
            print(f"Model saved to {model_save_path}")
    
    # Evaluation
    test_dataset = AmazonDataset(
        data_file="data/Subscription_Boxes.jsonl",
        processor=processor,
    )

    if os.path.exists("data/amazon_Subscription_Boxes_item_corpus.txt"):
        with open("data/amazon_Subscription_Boxes_item_corpus.txt", "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f.readlines()]
    else:
        print("Corpus file not found. Exiting evaluation.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: collate_fn(x, processor))
    if model_save_path:
        if args.peft:
            model = PixRecModel(args.model_name, is_peft = args.peft, model_save_path=model_save_path)
            # model.load_state_dict(torch.load(os.path.join(model_save_path, "full_model_state.pth")))
            state_dict = torch.load(os.path.join(model_save_path, "proj_model_state.pth"), map_location=device)
            
            with torch.no_grad():
                model.user_proj.load_state_dict(state_dict['user_proj'])
                model.item_proj.load_state_dict(state_dict['item_proj'])
                # model.user_proj.weight.copy_(state_dict['user_proj.weight'])
                # model.user_proj.bias.copy_(state_dict['user_proj.bias'])
                # model.item_proj.weight.copy_(state_dict['item_proj.weight'])
                # model.item_proj.bias.copy_(state_dict['item_proj.bias'])
            model = model.to(device)
        else:
            model = load_checkpoint_and_dispatch(model, model_save_path)
            # model.load_state_dict(torch.load(model_save_path))

    with torch.no_grad():
        evaluate(model, test_dataloader, corpus, processor, device)


if __name__ == "__main__":
    main()