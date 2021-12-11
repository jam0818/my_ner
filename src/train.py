import pdb

from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AdamW


def train(model: nn.Module,
          train_data_loader: DataLoader,
          optimizer: AdamW,
          scheduler,
          device,
          loss_type) -> nn.Module:

    total_loss = 0
    train_bar = tqdm(train_data_loader)

    for batch_idx, batch in enumerate(train_bar):
        batch_size = len(batch['input_ids'])
        # gpuに渡す時
        batch = {key: value.to(device) for key, value in batch.items()}

        # forward
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       token_type_ids=batch['token_type_ids'],
                       labels=batch['labels'])

        loss = output.loss['topic'] + output.loss['target']

        # backward
        if loss_type == 'total':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        elif loss_type == 'sep':
            optimizer.zero_grad()
            output.loss['topic'].backward(retain_graph=True)
            output.loss['topic'].backward()
            optimizer.step()
            scheduler.step()
        total_loss += (loss.item() * batch_size) / 2

        train_bar.set_postfix({
            'lr': scheduler.get_last_lr()[0],
            'loss': round(total_loss / (batch_idx + 1), 3)
        })

    total_loss = total_loss / len(train_data_loader.dataset)
    print(f'train_loss={total_loss}')

    return model
