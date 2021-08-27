import torch.nn as nn
import torch
from torch import optim
from torch.optim import Adam
import math
import time
from src.data import *
from src.Transformer import Transformer
from src.utils import initialize_weights, count_parameters, get_bleu, idx_to_word, epoch_time
from src.config import arg



model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=arg.d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=arg.max_len,
                    d_hid=arg.d_hid,
                    n_head=arg.n_heads,
                    n_layers=arg.n_layers,
                    dropout=arg.dropout,
                    device=arg.device).to(arg.device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=arg.init_lr,
                 weight_decay=arg.weight_decay,
                 eps=arg.adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=arg.factor,
                                                 patience=arg.patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
  model.train()
  epoch_loss = 0
  for i, batch in enumerate(iterator):
    src = batch.src
    trg = batch.trg

    optimizer.zero_grad()
    output = model(src, trg[:, :-1])
    output_reshape = output.contiguous().view(-1, output.shape[-1])
    trg = trg[:, 1:].contiguous().view(-1)

    loss = criterion(output_reshape, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()
    print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

  return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  batch_bleu = []
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      src = batch.src
      trg = batch.trg
      output = model(src, trg[:, :-1])
      output_reshape = output.contiguous().view(-1, output.shape[-1])
      trg = trg[:, 1:].contiguous().view(-1)

      loss = criterion(output_reshape, trg)
      epoch_loss += loss.item()

      total_bleu = []
      for j in range(arg.batch_size):
        try:
          trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
          output_words = output[j].max(dim=1)[1]
          output_words = idx_to_word(output_words, loader.target.vocab)
          bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
          total_bleu.append(bleu)
        except:
          pass

      total_bleu = sum(total_bleu) / len(total_bleu)
      batch_bleu.append(total_bleu)

  batch_bleu = sum(batch_bleu) / len(batch_bleu)
  return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
  train_losses, test_losses, bleus = [], [], []
  for step in range(total_epoch):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, arg.clip)
    valid_loss, bleu = evaluate(model, valid_iter, criterion)
    end_time = time.time()

    if step > arg.warmup:
      scheduler.step(valid_loss)

    train_losses.append(train_loss)
    test_losses.append(valid_loss)
    bleus.append(bleu)
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_loss:
      best_loss = valid_loss
      torch.save(model.state_dict(), 'model-{0}.pt'.format(valid_loss))

    print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
    print(f'\tBLEU Score: {bleu:.3f}')


run(total_epoch=arg.epoch, best_loss=arg.inf)