class Language:
    def __init__(self, name):
        self.name = name
        self.SOS_token = 0
        self.EOS_token = 1
        self.UNK_token = 2
        self.char2index = {"<sos>": self.SOS_token, "<eos>": self.EOS_token, "<unk>": self.UNK_token}
        self.char2count = {}
        self.index2char = {self.SOS_token: "<sos>", self.EOS_token: "<eos>", self.UNK_token: "<unk>"}
        self.n_tokens = 3  # Count SOS and EOS

    def addWord(self, word):
        for character in list(word):
            self.addCharacter(character)

    def addCharacter(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_tokens
            self.char2count[char] = 1
            self.index2char[self.n_tokens] = char
            self.n_tokens += 1
        else:
            self.char2count[char] += 1


def readLangs(filepath, lang_input, lang_target):

    # Read the file and split into lines
    lines = open(filepath, encoding="utf-8").read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split("\t")] for l in lines]

    input_lang = Language(lang_input)
    target_lang = Language(lang_target)

    return input_lang, target_lang, pairs


def prepareData(filepath, lang_input, lang_target):
    input_lang, target_lang, pairs = readLangs(filepath, lang_input, lang_target)

    for pair in pairs:
        input_lang.addWord(pair[0])
        target_lang.addWord(pair[1])
    return input_lang, target_lang, pairs
