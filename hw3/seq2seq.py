# inspired by https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb and https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb


import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, bidirectional, n_layers=1):
        super().__init__()

        self.bidirectional = bidirectional
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src_tensor):

        # src_tensor = [batch size, src len]

        embedded = self.dropout(self.embedding(src_tensor))

        # embedded = [batch size, src len, emb dim]

        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        if self.bidirectional:
            hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            cell = self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))

        return outputs, (hidden, cell)


class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, n_layers=1):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):

        # input = [batch size]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [batch size, n layers, hid dim]
        # context = [batch size, n layers, hid dim]

        input = input.unsqueeze(0)
        # input = [batch size, 1]

        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class AttentionDecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, n_layers=1):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = Attention(hid_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):

        # input = [batch size]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [batch size, n layers, hid dim]
        # context = [batch size, n layers, hid dim]

        input = input.unsqueeze(0)
        # input = [batch size, 1]

        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]

        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]

        lstm_input = torch.cat((embedded, weighted), dim=2)
        # lstm_input = [batch size, 1, (enc hid dim * 2) + emb dim]

        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), cell.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, attention):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src_tensor, trg_tensor, teacher_forcing_ratio=0.5):

        # src_tensor = [batch size, src len]
        # trg_tensor = [batch size, trg len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg_tensor.shape[0]
        trg_len = trg_tensor.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, (hidden, cell) = self.encoder(src_tensor)

        # first input to the decoder is the <sos> tokens
        input = trg_tensor[:, 0]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg_tensor[:, t] if teacher_force else top1

        return outputs
