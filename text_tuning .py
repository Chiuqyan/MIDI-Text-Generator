import argparse
import os
import random
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
#from torchviz import make_dot

import MIDI
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
import torch.nn as nn
#from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from MIdi_Cap import TextAndMIDIModel

EXTENSION = [".mid", ".midi"]
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] ='expandable_segments:True'


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


class MidiTextDataset(Dataset):
    def __init__(self, midi_list, tokenizer: MIDITokenizer,text_tokenizer, max_len=2048, min_file_size=3000, max_file_size=384000,
                 aug=True, check_alignment=True):

        self.tokenizer = tokenizer
        self.text_tokenizer = text_tokenizer
        self.midi_list = midi_list
        self.max_len = max_len
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.aug = aug
        self.check_alignment = check_alignment

    def __len__(self):
        return len(self.midi_list)

    def load_midi(self, index):
        path = self.midi_list[index]
        try:
            with open(path, 'rb') as f:
                datas = f.read()
            if len(datas) > self.max_file_size:  # large midi file will spend too much time to load
                raise ValueError("file too large")
            elif len(datas) < self.min_file_size:
                raise ValueError("file too small")
            mid = MIDI.midi2score(datas)
            if max([0] + [len(track) for track in mid[1:]]) == 0:
                raise ValueError("empty track")
            mid = self.tokenizer.tokenize(mid)
            if self.check_alignment and not self.tokenizer.check_alignment(mid):
                raise ValueError("not aligned")
            if self.aug:
                mid = self.tokenizer.augment(mid)
        except Exception:
            mid = self.load_midi(random.randint(0, self.__len__() - 1))
        return mid

    def get_text(self,midi_file_path):
        with open('/root/autodl-tmp/midi-model-main/MuCapsCaptions.json', 'r') as f:
            data = json.load(f)
        midi_file_name=os.path.basename(midi_file_path)
        text=None
        if midi_file_name in data:
            description = data[midi_file_name]["description"]
            style = data[midi_file_name]["style"]
            text = description + f"{description} The style is {style}."
            #print(text)
            
            text = self.text_tokenizer.encode(text)
            text = text[:self.max_len]
            text = torch.tensor(text)
            #text = text.clone().detach()
        return text

    def __getitem__(self, index):
        mid = self.load_midi(index)
        mid = np.asarray(mid, dtype=np.int16)
        # if mid.shape[0] < self.max_len:
        #     mid = np.pad(mid, ((0, self.max_len - mid.shape[0]), (0, 0)),
        #                  mode="constant", constant_values=self.tokenizer.pad_id)
        start_idx = random.randrange(0, max(1, mid.shape[0] - self.max_len))
        start_idx = random.choice([0, start_idx])
        mid = mid[start_idx: start_idx + self.max_len]
        mid = mid.astype(np.int64)
        mid = torch.from_numpy(mid)
        text = self.get_text(self.midi_list[index])

        #text = torch.tensor(text)

        return mid,text
def filter_midi_list(midi_list, json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    filtered_midi_list = [midi_file for midi_file in midi_list if os.path.basename(midi_file) in data]
    return filtered_midi_list

def collate_fn(batch,tokenizer,text_tokenizer):
    # 假设 batch 是一个元组列表，每个元组包含一对 MIDI 和文本
    midi_batch, text_batch = zip(*batch)

    # 对 MIDI 进行填充
    max_len_midi = max([len(mid) for mid in midi_batch])
    midi_batch = [F.pad(mid, (0, 0, 0, max_len_midi - mid.shape[0]), mode="constant", value=tokenizer.pad_id) for mid in midi_batch]
    midi_batch = torch.stack(midi_batch)

    # 对文本进行填充
    max_len_text = max([len(text) for text in text_batch])
    text_batch = [F.pad(text, (0, max_len_text - text.shape[0]), mode="constant", value=tokenizer.pad_id) for text in text_batch]
    text_batch = torch.stack(text_batch)

    return midi_batch, text_batch



def collate(batch):
    max_len = max([len(mid) for mid in batch])
    batch = [F.pad(mid, (0, 0, 0, max_len - mid.shape[0]), mode="constant", value=tokenizer.pad_id) for mid in batch]
    batch = torch.stack(batch)
    return batch


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def get_midi_list(path):
    all_files = {
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(path)
        for fname in files
    }
    all_midis = sorted(
        fname for fname in all_files if file_ext(fname) in EXTENSION
    )
    return all_midis


class TrainMIDIAndTextModel(TextAndMIDIModel):
    def __init__(self, tokenizer: MIDITokenizer, text_tokenizer,train_dataset,val_dataset, n_layer=12, n_head=16, n_embd=1024, n_inner=4096, flash=False,lr=2e-5, weight_decay=0.01, warmup=1e3, max_step=1e6):
        super(TrainMIDIAndTextModel, self).__init__(tokenizer=tokenizer, text_tokenizer=text_tokenizer, n_layer=n_layer, n_head=n_head, n_embd=n_embd,n_inner=n_inner, flash=flash)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_step = max_step
        self.text_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        #self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']  # no decay for bias and Norm
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay},
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        midi_x = batch[0][:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        text_x = batch[1][:, :-1].contiguous()  # (batch_size, text_sequence_length)
        midi_y = batch[0][:, 1:].contiguous()
        text_y = batch[1][:, 1:].contiguous()
        hidden = self.forward(midi_x, text_x)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        midi_y = midi_y.reshape(-1, midi_y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        text_y = text_y.reshape(-1, text_y.shape[-1])  # (batch_size*text_sequence_length)
        midi_x = midi_y[:, :-1]
        text_x = text_y[:, :-1]
        logits = self.forward_token(hidden, midi_x)
        base_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            midi_y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
            )
        #liner_weight = self.text_linear.weight
        #lsih=self.lstm.weight_ih_l0
        #lshh=self.lstm.weight_hh_l0
        #crossattention_weight =self.cross_attention.weight
        #l2_reg = torch.norm(liner_weight)**2 +torch.norm(lsih)**2+torch.norm(lshh)**2#+ torch.norm(crossattention_weight)**2
        #alpha=0.01
        loss=base_loss #+ alpha * l2_reg
        self.log("train/loss", loss)
        
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        #print(f"training loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        midi_x = batch[0][:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        text_x = batch[1][:, :-1].contiguous()  # (batch_size, text_sequence_length)
        midi_y = batch[0][:, 1:].contiguous()
        text_y = batch[1][:, 1:].contiguous()
        hidden = self.forward(midi_x, text_x)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        midi_y = midi_y.reshape(-1, midi_y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        text_y = text_y.reshape(-1, text_y.shape[-1])  # (batch_size*text_sequence_length)
        midi_x = midi_y[:, :-1]
        text_x = text_y[:, :-1]
        logits = self.forward_token(hidden_state=hidden, x=midi_x)
        base_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            midi_y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
            )
        #liner_weight = self.text_linear.weight
        #lsih=self.lstm.weight_ih_l0
        #lshh=self.lstm.weight_hh_l0
        #crossattention_weight =self.cross_attention.weight
        #l2_reg = torch.norm(liner_weight)**2+ +torch.norm(lsih)**2+torch.norm(lshh)**2 #+ torch.norm(crossattention_weight)**2
        #alpha=0.01
        loss=base_loss #+ alpha * l2_reg
        self.log("val/loss", loss, sync_dist=True)
        print(f"loss: {loss}")
        return loss
    
    def on_validation_start(self):
        torch.cuda.empty_cache()

    def on_validation_end(self):
        @rank_zero_only
        def gen_example():            
            mid = self.generate()
            mid = self.tokenizer.detokenize(mid)
            img = self.tokenizer.midi2img(mid)
            img.save(f"/root/autodl-fs/sample/{self.global_step}_0.png")
            with open(f"/root/autodl-fs/sample/{self.global_step}_0.mid", 'wb') as f:
                f.write(MIDI.score2midi(mid))
            prompt = val_dataset.load_midi(random.randint(0, len(val_dataset) - 1))
            prompt = np.asarray(prompt, dtype=np.int16)
            ori = prompt[:512]
            prompt = prompt[:256].astype(np.int64)
            mid = self.generate(prompt=prompt)
            mid = self.tokenizer.detokenize(mid)
            img = self.tokenizer.midi2img(mid)
            img.save(f"/root/autodl-fs/sample/{self.global_step}_1.png")
            img = self.tokenizer.midi2img(self.tokenizer.detokenize(ori))
            img.save(f"/root/autodl-fs/sample/{self.global_step}_1_ori.png")
            with open(f"/root/autodl-fs/sample/{self.global_step}_1.mid", 'wb') as f:
                f.write(MIDI.score2midi(mid))


            # Convert prompts to tensors
            #midi_prompt = np.asarray(midi_prompt, dtype=np.int16)
            #text_prompt = np.asarray(text_prompt, dtype=np.int16)
            #midi_prompt=None
            text_prompt="A cheerful and melodious pop song."
            text_promot=self.text_tokenizer(text_prompt)
            text_promot= torch.tensor(text_promot)
            text_prompt = text_prompt.unsqueeze(0)

            # Generate a MIDI sequence using text prompts
            mid = self.generate(text_prompt=text_prompt)#text_prompt=text_prompt
            mid = self.tokenizer.detokenize(mid)
            img = self.tokenizer.midi2img(mid)
            img.save(f"/root/autodl-fs/sample/{self.global_step}_2.png")
            with open(f"/root/autodl-fs/sample/{self.global_step}_2.mid", 'wb') as f:
                f.write(MIDI.score2midi(mid))
            

        try:
            gen_example()
        except Exception as e:
            print(f"error:{e}")
        torch.cuda.empty_cache()



'''def train_model(model, train_dataloader, val_dataloader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=len(train_dataloader) * num_epochs)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for mid, text in train_dataloader:
            mid = mid.to(device)
            text = text.to(device)

            optimizer.zero_grad()
            output = model(mid, text)
            loss = criterion(output, mid)
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        model.eval()
        
        with torch.no_grad():
            total_loss = 0
            for mid, text in val_dataloader:
                mid = mid.to(device)
                text = text.to(device)

                output = model(mid, text)
                loss = criterion(output, mid)
                total_loss += loss.item()

            avg_loss = total_loss / len(val_dataloader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss}')
            torch.cuda.empty_cache()

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()
        # 在每个epoch结束时，使用模型的generation函数生成一些样本
        sample_mid, sample_text = next(iter(val_dataloader))
        sample_mid = sample_mid.to(device)
        sample_text = sample_text.to(device)
        generated_output = model.generation(sample_mid, sample_text)
        print("Generated output: ", generated_output)

    return model'''


class CustomModelCheckpoint(ModelCheckpoint):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        #checkpoint = super().on_save_checkpoint(trainer, pl_module)

        # Save state_dicts of the parts you're interested in
        checkpoint['cross_attention_state_dict'] = pl_module.cross_attention.state_dict()
        checkpoint['text_linear_state_dict'] = pl_module.text_linear.state_dict()
        checkpoint['lstm_state_dict'] = pl_module.lstm.state_dict()  # assuming lstm is the name of your LSTM layer

        return checkpoint


if __name__ == '__main__':

    print("---load dataset---")
    midi_tokenizer = MIDITokenizer()

    text_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

        # 添加新的填充标记
    #text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    #text_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    midi_list = get_midi_list("/root/autodl-fs/unhashed")
    filtered_midi_list = filter_midi_list(midi_list, '/root/autodl-tmp/midi-model-main/MuCapsCaptions.json')
    random.shuffle(filtered_midi_list)
    full_dataset_len = len(filtered_midi_list)
    train_dataset_len = full_dataset_len - 128
    train_midi_list = filtered_midi_list[:train_dataset_len]
    val_midi_list = filtered_midi_list[train_dataset_len:]
    train_dataset = MidiTextDataset(train_midi_list, midi_tokenizer,text_tokenizer, max_len=1024)
    val_dataset = MidiTextDataset(val_midi_list, midi_tokenizer, text_tokenizer,max_len=1024, aug=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8, 
                                  num_workers=2,
                                  pin_memory=True,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.tokenizer,train_dataset.text_tokenizer))
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=8, 
                                shuffle=False,
                                persistent_workers=True,
                                num_workers=2,
                                pin_memory=True,
                                collate_fn=lambda batch: collate_fn(batch, train_dataset.tokenizer,train_dataset.text_tokenizer))

    print(f"train: {len(train_dataset)}  val: {len(val_dataset)}")

    # 初始化模型
    torch.cuda.empty_cache()
    #tb_logger = TensorBoardLogger('logs')
    model = TrainMIDIAndTextModel( #TextAndMIDIModel(
        tokenizer=midi_tokenizer,
        text_tokenizer=text_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_layer=12,
        n_head=16,
        n_embd=1024,
        n_inner=4096,
        flash=True,
        
    )
    model.load_model_weights(f"/root/autodl-fs/ckpt/last.ckpt")
    torch.cuda.empty_cache()
    checkpoint_callback =  CustomModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="align_epoch={epoch},loss={val/loss:.4f}",
        dirpath='/root/autodl-fs/ckpt',
    )
    callbacks = [checkpoint_callback]
    trainer = Trainer(
        precision= 16,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        accelerator="gpu",
        devices=-1,
        max_steps=1e6,
        benchmark=True,
        val_check_interval=100,
        log_every_n_steps=10,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        default_root_dir='/root/autodl-tmp/midi-model-main',
    )
    
    '''model_graph_filename = 'model_graph.pdf'
    if not os.path.exists(model_graph_filename):
        # 如果文件不存在，那么生成模型结构图

        # 从数据加载器中获取一个批次的数据
        x, text_input = next(iter(train_dataloader))

        # 将数据移动到模型所在的设备
        x = x.to(model.device)
        text_input = text_input.to(model.device)

        # 将数据通过模型进行一次前向传播
        y = model(x, text_input)

        # 生成模型的计算图
        dot = make_dot(y, params=dict(model.named_parameters()))

        # 保存为PDF文件
        dot.format = 'pdf'
        dot.render(filename=model_graph_filename)'''
    
    print("---start train---")
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloader, val_dataloader)#, ckpt_path=ckpt_path)
    trainer.save_checkpoint()

    #train_model(model, train_dataloader, val_dataloader)

    # 加载最佳模型并导出为ONNX
    '''model.load_state_dict(torch.load('best_model.pth'))
    dummy_x = torch.randn(32, 4096, 1024)

    # 创建dummy文本序列
    # 假设你的文本tokenizer的词汇表大小为vocab_size
    vocab_size = text_tokenizer.vocab_size
    dummy_text_input = torch.randint(low=0, high=vocab_size, size=(32, 512))

    dummy_input = (dummy_x, dummy_text_input)
    torch.onnx.export(model, dummy_input, "best_gen_model.onnx")'''


