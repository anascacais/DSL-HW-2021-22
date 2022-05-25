## inspired in https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350
# and https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# # built-in
import argparse
import time

# third-party
from sklearn.metrics import accuracy_score
import torch
from torch import optim

# local
from language import prepareData
from seq2seq2 import DecoderLSTM, EncoderLSTM, Seq2Seq
from utils import configure_device, configure_seed, plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSequence(lang, word):
    return [
        lang.char2index[char] if char in lang.char2index.keys() else lang.char2index["<unk>"] for char in list(word)
    ]


def tensorFromSequence(lang, word, opt):
    indexes = [lang.SOS_token]
    if opt.invert_src:
        indexes += list(reversed(indexesFromSequence(lang, word)))
    else:
        indexes += indexesFromSequence(lang, word)
    indexes.append(lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(opt.batch_size, -1)


def tensorsFromPair(pair, input_lang, target_lang, opt):
    input_tensor = tensorFromSequence(input_lang, pair[0], opt)
    target_tensor = tensorFromSequence(target_lang, pair[1], opt)
    return (input_tensor, target_tensor)


def transliteration(lang, tensor):

    list_chars = [
        lang.index2char[i.item()] if i.item() in lang.index2char.keys() else lang.index2char[lang.UNK_token]
        for i in tensor[0]
    ]

    return "".join(list_chars)


def evaluate(model, input_tensor, target_tensor):

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor, target_tensor)
        output_tensor = output_tensor[:, 1:, :].view(-1, output_tensor.shape[-1]).argmax(1)
        target_tensor = target_tensor[:, 1:].view(-1)

        return accuracy_score(target_tensor, output_tensor)


def train(model, input_tensor, target_tensor, optimizer, clip=1):

    # input_tensor = [batch size, src len]
    # targer_tensor = [batch size, trg len]

    # Clear the accumulating gradients
    optimizer.zero_grad()

    # Pass the input and target for model's forward method
    output_tensor = model(input_tensor, target_tensor)
    # output_tensor = [batch_size, trg length, output dim]

    output_tensor = output_tensor[:, 1:, :].view(
        -1, output_tensor.shape[-1]
    )  # remove first token corresponding to <sos>
    target_tensor = target_tensor[:, 1:].view(-1)
    # target_tensor = [(trg len - 1) * batch size]
    # output_tensor = [(trg len - 1) * batch size, output dim]

    loss = model.criterion(output_tensor, target_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return loss


def train_epochs(model, input_lang, target_lang, train_pairs, eval_pairs, test_pairs, opt):

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    train_mean_losses = []
    valid_mean_accs = []
    valid_accs = []
    test_accs = []
    train_losses = []
    start_time = time.time()

    training_pairs = [tensorsFromPair(train_pairs[i], input_lang, target_lang, opt) for i in range(len(train_pairs))]
    evaluation_pairs = [tensorsFromPair(eval_pairs[i], input_lang, target_lang, opt) for i in range(len(eval_pairs))]
    testing_pairs = [tensorsFromPair(test_pairs[i], input_lang, target_lang, opt) for i in range(len(test_pairs))]

    for ii in range(1, opt.epochs + 1):
        print("Training epoch {}".format(ii))
        model.train()
        for i, (input_tensor, target_tensor) in enumerate(training_pairs):  # batch_size = 1
            print("{} of {}".format(i + 1, len(training_pairs)), end="\r")
            loss = train(model, input_tensor, target_tensor, optimizer)
            train_losses.append(loss.item())

        mean_loss = torch.tensor(train_losses).mean().item()
        print("Training loss: %.4f" % mean_loss)

        train_mean_losses.append(mean_loss)

        valid_accs = []
        for i, (input_tensor, target_tensor) in enumerate(evaluation_pairs):
            acc = evaluate(model, input_tensor, target_tensor)
            valid_accs.append(acc)
        valid_mean_accs.append(sum(valid_accs) / len(valid_accs))
        print("Valid acc: %.4f" % (valid_mean_accs[-1]))

    run_time = time.time() - start_time
    print(f"Training time: {run_time}")

    for i, (input_tensor, target_tensor) in enumerate(testing_pairs):
        acc = evaluate(model, input_tensor, target_tensor)
        test_accs.append(acc)
    print("Final Test acc: %.4f" % (sum(test_accs) / len(test_accs)))

    str_epochs = [str(i) for i in range(1, opt.epochs + 1)]

    if opt.invert_src:
        plot(str_epochs, train_mean_losses, ylabel="Loss", name=f"training-loss-inv-{opt.attention}")
        plot(str_epochs, valid_mean_accs, ylabel="Accuracy", name=f"validation-accuracy-inv-{opt.attention}")
    else:
        plot(str_epochs, train_mean_losses, ylabel="Loss", name=f"training-loss-{opt.attention}")
        plot(str_epochs, valid_mean_accs, ylabel="Accuracy", name=f"validation-accuracy-{opt.attention}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-hidden_size", type=int, default=125)
    parser.add_argument("-embd_size", type=int, default=256)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-l2_decay", type=float, default=0.0)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-gpu_id", type=int, default=None)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-attention", type=int, default=True)
    parser.add_argument("-invert_src", type=int, default=True)
    opt = parser.parse_args()

    configure_seed(opt.seed)
    configure_device(opt.gpu_id)

    input_lang, target_lang, train_pairs = prepareData("ar2en-train.txt", "ar", "en")
    _, _, test_pairs = prepareData("ar2en-test.txt", "ar", "en")
    _, _, eval_pairs = prepareData("ar2en-eval.txt", "ar", "en")

    print(f"Vocabulary sizes: ({input_lang.name}) {input_lang.n_tokens} | ({target_lang.name}) {target_lang.n_tokens}")

    encoder = EncoderLSTM(input_lang.n_tokens, opt.embd_size, opt.hidden_size, opt.dropout, opt.attention).to(device)
    decoder = DecoderLSTM(target_lang.n_tokens, opt.embd_size, opt.hidden_size, opt.dropout, opt.attention).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    train_epochs(model, input_lang, target_lang, train_pairs, eval_pairs, test_pairs, opt)


if __name__ == "__main__":
    main()
