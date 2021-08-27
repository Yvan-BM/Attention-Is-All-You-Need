from src.Transformer import Transformer
from src.utils import initialize_weights


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    d_hid=d_hid,
                    n_head=n_heads,
                    n_layers=n_layers,
                    dropout=dropout,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

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
      for j in range(batch_size):
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