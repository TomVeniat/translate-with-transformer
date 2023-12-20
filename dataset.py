import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


def causal_mask(size):
    mask = torch.ones(size, size)
    mask = torch.triu(mask, diagonal=1, ).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)//100

    def __getitem__(self, item):
        src_tgt_pair = self.ds[item]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_n_pad = self.seq_len - len(enc_input_tokens) - 2
        dec_n_pad = self.seq_len - len(dec_input_tokens) - 1

        if enc_n_pad < 0 or dec_n_pad < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_n_pad, dtype=torch.int64)
            ], dim=0
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_n_pad, dtype=torch.int64)
            ], dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_n_pad, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        enc_mask = (encoder_input != self.pad_token).unsqueeze(0).int()
        dec_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(
            decoder_input.size(0))
        # We prefer (B, 1, seq_len) to (B, 1, 1, seq_len) because we split the heads in the MH attention
        # enc_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        # dec_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(
        #     decoder_input.size(0))

        return dict(
            encoder_input=encoder_input,  # (seq_len)
            decoder_input=decoder_input,  # (seq_len)
            encoder_mask=enc_mask,  # (1, 1, seq_len)
            decoder_mask=dec_mask,  # (1, seq_len) & (1, seq_len, seq_len) -> (1, seq_len, seq_len)
            label=label,  # (seq_len)
            src_text=src_text,
            tgt_text=tgt_text
        )
