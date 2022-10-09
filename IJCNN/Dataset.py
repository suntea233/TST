import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer,BertTokenizer
import pandas as pd

max_len = 1024
class IMDBDataset(Dataset):
    def __init__(self,path):
        super(IMDBDataset, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(r"C:\Attention\BART\bartbase")
        self.negative,self.positive = self.preprocessing(path)

    def __len__(self):

        return min(len(self.negative),len(self.positive))

    def __getitem__(self, item):
        return torch.tensor(self.negative[item]),torch.tensor(self.positive[item])

    def preprocessing(self,path):

        data = pd.read_csv(path)
        positive = []
        negative = []

        for i in range(len(data['sentence'])):
            if data['label'][i] == "positive":
                positive.append(self.tokenizer(data['sentence'][i],truncation=True)['input_ids'])
            else:
                negative.append(self.tokenizer(data['sentence'][i],truncation=True)['input_ids'])
            # if count > 10000:
            #     return negative,positive
        return negative, positive


    def get_attention_mask(self,inputs):
        attention_mask = []
        for input_id in inputs:
            temp = [0 for _ in range(len(input_id))]
            for index,value in enumerate(input_id):
                if value == 1:
                    temp[index] = 1
            attention_mask.append(temp)
        return attention_mask

    def collate_fn(self,batch):
        positive_inputs = []
        negative_inputs = []
        positive_outputs = []
        negative_outputs = []
        for n,p in batch:
            negative_inputs.append(n[:-1])
            positive_inputs.append(p[:-1])
            negative_outputs.append(n[1:])
            positive_outputs.append(p[1:])
        positive_inputs = pad_sequence(positive_inputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)
        negative_inputs = pad_sequence(negative_inputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)

        positive_outputs = pad_sequence(positive_outputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)
        negative_outputs = pad_sequence(negative_outputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)

        positive_outputs_attention_mask = self.get_attention_mask(positive_outputs)
        negative_outputs_attention_mask = self.get_attention_mask(negative_outputs)

        positive_inputs_attention_mask = self.get_attention_mask(positive_inputs)
        negative_inputs_attention_mask = self.get_attention_mask(negative_inputs)

        return positive_inputs,\
               negative_inputs,\
               torch.tensor(positive_inputs_attention_mask),\
               torch.tensor(negative_inputs_attention_mask),\
               positive_outputs,\
               negative_outputs,\
               torch.tensor(positive_outputs_attention_mask),\
               torch.tensor(negative_outputs_attention_mask)


class SSTDataset(Dataset):
    def __init__(self,path):
        super(SSTDataset, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(r"C:\Attention\BART\bartbase")
        self.positive,self.negative = self.preprocessing(path)

    def __len__(self):
        return min(len(self.positive),len(self.negative))

    def __getitem__(self, item):
        return torch.tensor(self.positive[item]),torch.tensor(self.negative[item])

    def preprocessing(self,path):
        positive = []
        negative = []
        data = pd.read_csv(
            path,
            sep='\t',
        )

        for i in range(len(data['sentence'])):
            if data['label'][i] == 1:
                positive.append(self.tokenizer(data['sentence'][i],truncation=True,max_length=max_len)['input_ids'])
            else:
                negative.append(self.tokenizer(data['sentence'][i],truncation=True,max_length=max_len)['input_ids'])
            # if count > 10000:
            #     return negative,positive

        return negative, positive


    def get_attention_mask(self,inputs):
        attention_mask = []
        for input_id in inputs:
            temp = [1 for _ in range(len(input_id))]
            for index,value in enumerate(input_id):
                if value == 1:
                    temp[index] = 0
            attention_mask.append(temp)
        return attention_mask


    def collate_fn(self,batch):
        positive_inputs = []
        negative_inputs = []
        positive_outputs = []
        negative_outputs = []
        for n,p in batch:
            negative_inputs.append(n[:-1])
            positive_inputs.append(p[:-1])
            negative_outputs.append(n[1:])
            positive_outputs.append(p[1:])
        positive_inputs = pad_sequence(positive_inputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)
        negative_inputs = pad_sequence(negative_inputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)

        positive_outputs = pad_sequence(positive_outputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)
        negative_outputs = pad_sequence(negative_outputs,padding_value=self.tokenizer.pad_token_id).transpose(0, 1)

        positive_outputs_attention_mask = self.get_attention_mask(positive_outputs)
        negative_outputs_attention_mask = self.get_attention_mask(negative_outputs)

        positive_inputs_attention_mask = self.get_attention_mask(positive_inputs)
        negative_inputs_attention_mask = self.get_attention_mask(negative_inputs)

        return positive_inputs,\
               negative_inputs,\
               torch.tensor(positive_inputs_attention_mask),\
               torch.tensor(negative_inputs_attention_mask),\
               positive_outputs,\
               negative_outputs,\
               torch.tensor(positive_outputs_attention_mask),\
               torch.tensor(negative_outputs_attention_mask)


# import time
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# now = time.time()
# dataset = SSTDataset(r"C:\Attention\SST-2\train.tsv")
# # train,test = train_test_split(sentences,test_size=0.2)
# dataloader = DataLoader(dataset, batch_size=2, num_workers=0,collate_fn=dataset.collate_fn)
# # input_ids,labels,attention_mask =
# count = 0
# for positive, negative, positive_attention_mask, negative_attention_mask, positive_outputs, negative_outputs, positive_outputs_attention_maks, negative_outputs_attention_mask in dataloader:
#     positive, negative, positive_attention_mask, negative_attention_mask, positive_outputs, negative_outputs, positive_outputs_attention_maks, negative_outputs_attention_mask = positive.to(
#         DEVICE), negative.to(DEVICE), positive_attention_mask.to(DEVICE), negative_attention_mask.to(
#         DEVICE), positive_outputs.to(DEVICE), negative_outputs.to(DEVICE), positive_outputs_attention_maks.to(
#         DEVICE), negative_outputs_attention_mask.to(DEVICE)
#
# print(time.time()-now)