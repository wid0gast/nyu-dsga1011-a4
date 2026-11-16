import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
P_SYNONYM = 0.25
P_TYPO = 0.25
P_CAPS = 0.05

KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def synonym_replace(word):
    if not word.isalpha() or len(word) < 4:
        return word
    
    candidates = set()
    for synset in wordnet.synsets(word.lower())[:1]:
        for lemma in synset.lemmas():
            name = lemma.name()
            if "_" not in name and name.lower() != word.lower() and name.isalpha() and len(name) > 3:
                candidates.add(name)
    if not candidates:
        return word
    new_word = random.choice(sorted(candidates))
    if word[0].isupper():
        new_word = new_word.capitalize()
    return new_word

def add_typo(word):
    if not word.isalpha() or len(word) < 3:
        return word
    i = random.randint(1, len(word)-1)
    typo = random.choice(["sub", "drop", "repeat", "swap"])

    if typo == "sub":
        neighbors = KEYBOARD_NEIGHBORS.get(word[i].lower())
        if not neighbors:
            return word
        new_char = random.choice(neighbors)
        if word[i].isupper():
            new_char = new_char.upper()
        return word[:i] + new_char + word[i+1:]
    elif typo == "drop":
        return word[:i] + word[i+1:]
    elif typo == "repeat":
        return word[:i] + word[i] + word[i:]
    else:
        if i == len(word) - 1:
            i -= 1
        return word[:i] + word[i+1] + word[i] + word[i+2:]

def random_caps(word):
    if word.isalpha():
        return word.upper()
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    words = word_tokenize(example["text"].replace("<br />", " "))
    new_words = []
    for word in words:
        if random.random() < P_SYNONYM:
            word = synonym_replace(word)
        if random.random() < P_TYPO:
            word = add_typo(word)
        if random.random() < P_CAPS:
            word = random_caps(word)
        new_words.append(word)

    example["text"] = re.sub(r" ([.,!?;:])", r"\1", TreebankWordDetokenizer().detokenize(new_words))

    ##### YOUR CODE ENDS HERE ######

    return example
