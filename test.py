import torch
from src.utils import idx_to_word, get_bleu, count_parameters
from src.Transformer import Transformer
from src.data import *
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
                    dropout=0.00,
                    device=arg.device).to(arg.device)

print(f'The model has {count_parameters(model):,} trainable parameters')


def test_model(num_examples):
  iterator = test_iter
  model.load_state_dict(torch.load("model-saved.pt"))

  with torch.no_grad():
    batch_bleu = []
    for i, batch in enumerate(iterator):
      src = batch.src
      trg = batch.trg
      output = model(src, trg[:, :-1])

      total_bleu = []
      for j in range(num_examples):
        try:
          src_words = idx_to_word(src[j], loader.source.vocab)
          trg_words = idx_to_word(trg[j], loader.target.vocab)
          output_words = output[j].max(dim=1)[1]
          output_words = idx_to_word(output_words, loader.target.vocab)

          print('source :', src_words)
          print('target :', trg_words)
          print('predicted :', output_words)
          print()
          bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
          total_bleu.append(bleu)
        except:
          pass

      total_bleu = sum(total_bleu) / len(total_bleu)
      print('BLEU SCORE = {}'.format(total_bleu))
      batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    print('TOTAL BLEU SCORE = {}'.format(batch_bleu))

test_model(num_examples=arg.batch_size)