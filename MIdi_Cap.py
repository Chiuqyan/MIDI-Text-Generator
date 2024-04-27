from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
import torch
#from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer, BertTokenizer,AutoModelForMaskedLM
from transformers import GPT2LMHeadModel
import os


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output
class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck_dim,dropout_p=0.2):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_dim, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, out_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TextAndMIDIModel(pl.LightningModule):
    def __init__(self, tokenizer: MIDITokenizer,text_tokenizer, n_layer=12, n_head=16, n_embd=1024, n_inner=4096, flash=False,
                 *args, **kwargs):
        super(TextAndMIDIModel, self).__init__()
    
        
        
        self.tokenizer = tokenizer
        self.midi_model = MIDIModel(tokenizer)
        self.midi_model.load_state_dict(torch.load('Yeerchiu/midi-llama-test1/model.ckpt'), strict=False)# midi ckpt path here
        for param in self.midi_model.parameters():  # frozen
            param.requires_grad = False
        self.text_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.text_tokenizer=text_tokenizer
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_linear = Bottleneck(50257, n_embd,512)#nn.Linear(50257, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=2, batch_first=True)
        self.cross_attention = CrossAttention(n_embd, n_head)

    def forward_token(self, hidden_state, x=None):
        hidden_state = hidden_state.unsqueeze(1)  # (batch_size, 1, n_embd)
        if x is not None:
            x = self.midi_model.net_token.embed_tokens(x)
            hidden_state = torch.cat([hidden_state, x], dim=1)
        hidden_state = self.midi_model.net_token.forward(inputs_embeds=hidden_state).last_hidden_state
        return self.midi_model.lm_head(hidden_state)




    def forward(self, x, text_input=None):
        # process text input with BERT
        text_output = self.text_model(text_input)[0]  # we only need the last hidden state
        text_output = self.text_linear(text_output)  # convert tokenized output dimension to n_embd
        text_output, _ = self.lstm(text_output)
        #print("text_out")
        #print(text_output.shape)


        # ensure text_output and x have the same dimensions
        midi_output = self.midi_model(x)
        #print("midi_out")
        #print(midi_output.shape)
        
        cross_attn_output = self.cross_attention(midi_output.transpose(0, 1), text_output.transpose(0, 1), text_output.transpose(0, 1))
        # merge token sequence
        x = self.midi_model.net.embed_tokens(x)
        x = x.sum(dim=-2)
        cross_attn_output = cross_attn_output.transpose(0, 1)

        #print(x.shape)
        #print(cross_attn_output.shape)
        x = x + cross_attn_output


        #x = torch.cat([x, text_output], dim=-2)  # concatenate text output with MIDI embeddings
        x = self.midi_model.net.forward(inputs_embeds=x)
        return x.last_hidden_state
    def load_model_weights(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):  # check if file exists
            checkpoint = torch.load(checkpoint_path)

            # Check if the keys exist in the checkpoint before loading
            if 'cross_attention_state_dict' in checkpoint:
                self.cross_attention.load_state_dict(checkpoint['cross_attention_state_dict'])
            if 'text_linear_state_dict' in checkpoint:
                self.text_linear.load_state_dict(checkpoint['text_linear_state_dict'])
            if 'lstm_state_dict' in checkpoint:
                self.lstm.load_state_dict(checkpoint['lstm_state_dict'])  # assuming lstm is the name of your LSTM layer
        else:
            print(f"No checkpoint found at {checkpoint_path}")


    def sample_top_p_k(self, probs, p, k):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        mask = torch.zeros(probs_sort.shape[-1], device=probs_sort.device)
        mask[:k] = 1
        probs_sort = probs_sort * mask
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        shape = probs_sort.shape
        next_token = torch.multinomial(probs_sort.reshape(-1, shape[-1]), num_samples=1).reshape(*shape[:-1], 1)
        next_token = torch.gather(probs_idx, -1, next_token).reshape(*shape[:-1])
        return next_token

    @torch.inference_mode()
    def generate(self, prompt=None, text_prompt=None,max_len=512, temp=1.0, top_p=0.98, top_k=20, amp=True):
        tokenizer = self.tokenizer
        max_token_seq = tokenizer.max_token_seq
        if prompt is None:
            input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=self.device)
            input_tensor[0, 0] = tokenizer.bos_id  # bos
        else:
            prompt = prompt[:, :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                                mode="constant", constant_values=tokenizer.pad_id)
            input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=self.device)
        #

        if text_prompt is not None:

            text_prompt = text_prompt.unsqueeze(0)
            text_prompt = text_prompt[:,:max_token_seq]
        
            if text_prompt.shape[-1] < max_token_seq:
                text_prompt = np.pad(text_prompt, ((0, 0), (0, max_token_seq - text_prompt.shape[-1])),mode="constant",
                                     constant_values=tokenizer.pad_id,device=self.device)
            text_prompt = torch.tensor(text_prompt).to(dtype=torch.long, device=model.device)

        else:
            text_prompt = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=self.device)
            text_prompt[0, 0] = tokenizer.bos_id  # bos#self.text_tokenizer.bos_token_id
        #text_prompt = torch.tensor(text_prompt).to(dtype=torch.long, device=self.device)
        #text_prompt = text_prompt.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input padding ok")

        cur_len = input_tensor.shape[1]
        bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
        with bar, torch.cuda.amp.autocast(enabled=amp):
            while cur_len < max_len:
                end = False
                hidden = self.forward(input_tensor,text_prompt)[0, -1].unsqueeze(0)
                #print("hidden forwarded")
                next_token_seq = None
                event_name = ""
                for i in range(max_token_seq):
                    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.int64, device=self.device)
                    if i == 0:
                        mask[list(tokenizer.event_ids.values()) + [tokenizer.eos_id]] = 1
                    else:
                        param_name = tokenizer.events[event_name][i - 1]
                        mask[tokenizer.parameter_ids[param_name]] = 1

                    logits = self.forward_token(hidden, next_token_seq)[:, -1:]
                    #print("token forwarded")
                    scores = torch.softmax(logits / temp, dim=-1) * mask
                    sample = self.sample_top_p_k(scores, top_p, top_k)
                    if i == 0:
                        next_token_seq = sample
                        eid = sample.item()
                        if eid == tokenizer.eos_id:
                            end = True
                            break
                        event_name = tokenizer.id_events[eid]
                    else:
                        next_token_seq = torch.cat([next_token_seq, sample], dim=1)
                        #print("next token seq cat")
                        if len(tokenizer.events[event_name]) == i:
                            break
                if next_token_seq.shape[1] < max_token_seq:
                    next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                           "constant", value=tokenizer.pad_id)
                next_token_seq = next_token_seq.unsqueeze(1)
        


                input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
                cur_len += 1
                bar.update(1)
                if end:
                    break
        return input_tensor[0].cpu().numpy()
