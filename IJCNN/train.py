from diff.IJCNN.Dataset import SSTDataset
from model import Encoder,Decoder
from diffusion import Diffusion

import torch
import tqdm
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SSTDataset(r"../input/diffusion-dataset/train.tsv")
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=0, collate_fn=train_dataset.collate_fn)
print("data load complete")

encoder = Encoder()
decoder = Decoder()
diffusion = Diffusion(encoder, decoder, train_dataset.tokenizer.pad_token_id)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
epochs = 10
print("model initial complete")

# sample_example = "The weather today is good!"
# sample_example = train_dataset.tokenizer(sample_example)['input_ids']
# sample_example = torch.tensor(sample_example)
# sample_example = sample_example.unsqueeze(0)
# sample_example_attention_mask = torch.zeros_like(sample_example)


for i in tqdm.tqdm(range(epochs)):
    diffusion.train()
    train_loss = 0
    lm_losses = 0
    diffusion_losses = 0
    count = 0
    reconstruction_losses = 0

    #     if i % 5 == 0:
    #         xT = sample(sample_example,sample_example_attention_mask)
    #         print(xT)
    diffusion.train()
    for positive, negative, positive_attention_mask, negative_attention_mask, positive_outputs, negative_outputs, positive_outputs_attention_maks, negative_outputs_attention_mask in train_dataloader:
        count += 1

        positive, negative, positive_attention_mask, negative_attention_mask, positive_outputs, negative_outputs, positive_outputs_attention_maks, negative_outputs_attention_mask = positive.to(
            DEVICE), negative.to(DEVICE), positive_attention_mask.to(DEVICE), negative_attention_mask.to(
            DEVICE), positive_outputs.to(DEVICE), negative_outputs.to(DEVICE), positive_outputs_attention_maks.to(
            DEVICE), negative_outputs_attention_mask.to(DEVICE)
        lm_loss, diffusion_loss = diffusion(positive, negative, positive_attention_mask, negative_attention_mask,
                                            positive_outputs, negative_outputs)

        loss = lm_loss + diffusion_loss
        lm_losses += lm_loss.item()
        diffusion_losses += diffusion_loss.item()
        #         reconstruction_losses += reconstruction_loss.item()
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #     print("train_loss = {}, lm_ loss = {}, diffusion_loss = {}, reconstruction_loss = {}".format((train_loss / count), (lm_losses / count),
    #                                                                       (diffusion_losses / count),(reconstruction_loss) / count))

    print("train_loss = {}, lm_loss = {}, diffusion_loss = {}".format((train_loss / count), (lm_losses / count),
                                                                      (diffusion_losses / count)))


def sample(x0, x0_attention_mask):
    last_decode_logits = diffusion(x0, x0_attention_mask)
    xT_ids = torch.argmax(last_decode_logits, dim=-1)
    xT = train_dataset.tokenizer.convert_ids_to_tokens(xT_ids)
    return xT


dev_dataset = SSTDataset(r"../input/diffusion-dataset/dev.tsv")
dev_dataloader = DataLoader(dev_dataset, batch_size=1, num_workers=0, collate_fn=dev_dataset.collate_fn)

print(diffusion.a)
for positive, negative, positive_attention_mask, negative_attention_mask, positive_outputs, negative_output, positive_outputs_attention_mask, negative_outpus_attention_mask in dev_dataloader:
    s = torch.tensor([[train_dataset.tokenizer.bos_token_id]]).to(DEVICE)
    positive, positive_attention_mask = positive.to(DEVICE), positive_attention_mask.to(DEVICE)
    diffusion.eval()
    flag = True
    data = torch.tensor([]).long().unsqueeze(0).to(DEVICE)
    count = 0
    while flag:
        s = torch.cat((s, data), dim=-1)
        #         print(s)
        outputs = diffusion.sample(s, positive)
        #         print(outputs)
        prob = outputs.max(dim=-1, keepdim=False)[1]

        data = prob[-1].unsqueeze(0).unsqueeze(0)
        #         print(data)
        count += 1
        if data == train_dataset.tokenizer.eos_token_id:
            flag = False
        if count == 50:
            flag = False
    print(train_dataset.tokenizer.decode(s[-1]))


