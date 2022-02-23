from logging import raiseExceptions
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from collections import defaultdict, Counter

'''
get_prediction loads 'model' state dict and vocab files > vocab_size, label_size, index of PAD
'''
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, filter_heights, stride, dropout, num_classes, pad_idx):
        super(CNN, self).__init__()
        
        # Create an embedding layer (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        #   to represent the words in vocabulary. Make sure to use vocab_size, embed_size, and pad_idx here.

        self.embeddinglayer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=pad_idx)

        # Define multiple Convolution layers (nn.Conv2d) with filter (kernel) size [filter_height, embed_size] based on 
        #   different filter_heights.
        # Input channels will be 1 and output channels will be out_channels (these many different filters will be trained 
        #   for each convolution layer)
        # Note: even though conv layers are nn.Conv2d, we are doing a 1d convolution since we are only moving the filter 
        #   in one direction
          # kernel_size - (height, width)

        self.convlayers = nn.ModuleList(
          [nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(i, embed_size), stride=stride) for i in filter_heights]
        )
        self.relu = nn.ReLU()

        # Create a dropout layer (nn.Dropout) using dropout
        
        self.dropoutlayer = nn.Dropout(p=dropout, inplace=False)

        # Define a linear layer (nn.Linear) that consists of num_classes units 
        #   and takes as input the concatenated output for all cnn layers (out_channels * num_of_cnn_layers units)
        # length of filter_heights equals the number of layers, marc=out_channels*layers
        
        self.linearlayer = nn.Linear(in_features=out_channels*len(filter_heights), out_features=num_classes, bias=True)


    def forward(self, texts):
        """
        texts: LongTensor [batch_size, max_len]
        
        Returns output: Tensor [batch_size, num_classes]
        """
        # Pass texts through embedding layer to convert from word ids to word embeddings
        #   Resulting: shape: [batch_size, max_len, embed_size]
        
        x = self.embeddinglayer(texts)
        #print('after embed [batch_size, max_len, embed_size]:', x.shape)

        # Input to conv should have 1 channel.
        #   Resulting shape: [batch_size, 1, MAX_LEN, embed_size]
        
        x = torch.unsqueeze(x, 1)
        #print('after unsqueeze [batch_size, 1, MAX_LEN, embed_size]:', x.shape)
        
        # Pass these texts to each conv layer and compute their output as follows:
        #   Output will have shape [batch_size, out_channels, *, 1] where * depends on filter_height and stride
        #   Convert to shape [batch_size, out_channels, *] (see torch's squeeze() function)
        #   Apply non-linearity on it 
        #   Take the max value across last dimension to have shape [batch_size, out_channels]
        # Concatenate (torch.cat) outputs from all cnns [batch_size, (out_channels*num_of_cnn_layers)]
        #

        results = [None] * len(self.convlayers)
        for i, conv in enumerate(self.convlayers):
          results[i] = conv(x)
          #print('after conv [batch_size, out_channels, *, 1]:', results[i].shape)
          results[i] = torch.squeeze(results[i], dim=3)
          #print('after squeezed:', results[i].shape)
          results[i] = self.relu(results[i])
          #print('after relu:', results[i].shape)
          results[i] = torch.max(results[i], dim=2)[0]
          #print('after max:', results[i].shape)
          #results[i] = torch.squeeze(results[i])
          #print('after squeezed:', results[i].shape)
        x = torch.cat(results, dim=1)
        #print('after cat [batch_size, (out_channels*num_of_cnn_layers)]:', x.shape)

        #   Since each cnn is of different filter_height, it will look at different number of words at a time
        #     So, a filter_height of 3 means cnn looks at 3 words (3-grams) at a time and tries to extract some information from it
        #   Each cnn will learn out_channels number of features from the words it sees at a time
        #   Then applied a non-linearity and took the max value for all channels
        #    Essentially trying to find important n-grams from the entire text
        # Everything happens on a batch simultaneously hence that additional batch_size as the first dimension

        # Apply dropout
        
        x = self.dropoutlayer(x)
        #print('after dropout:', x.shape)

        # Pass output through the linear layer and return its output 
        #   Resulting shape: [batch_size, num_classes]
        # (((W - K + 2P)/S) + 1)

        x = self.linearlayer(x)
        #print('after linear [batch_size, num_classes]:', x.shape)

        ##### NOTE: Do not apply a sigmoid or softmax to the final output - done in training method!

        return x


class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None, label2idx=None):

        self.examples = examples # pair of str(label), list(description)
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.label2idx = label2idx
        if split == 'train':
          self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        self.label_size = len(self.label2idx)
      
        # Convert text to indices
        self.textual_ids = []
        self.label_ids = []
        self.convert_text()
        self.convert_labels()
    
        # Convert back to text
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def build_dictionary(self): 
        '''
        Builds the dictionaries idx2word, word2idx and a label 'dictionary' label2idx. 
        This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. Uses self.threshold
        to control which words are assigned indices in the dictionaries.
        Returns nothing.
        '''
        assert self.split == 'train'
        
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}
        self.label2idx = {UNK:0}
        self.idx2label = {0:UNK}

        # Count the frequencies of all words in the training data (self.examples)
        # Assign idx (starting from 3) to all words having word_freq >= self.threshold
        # Convert to uppercase
        words = [word.upper() for (_, desc) in self.examples for word in desc if word != None]

        rawcounts = Counter(words)
        adjcounts = rawcounts - Counter(dict.fromkeys(rawcounts, self.threshold - 1)) #Subtract same keys with counts equal to threshold
        +adjcounts # Removes all zero count keys, hence the -1 above

        try:
          test = set([label for label, desc in self.examples])
        except:
          print(test)
        idx2wordadder = dict(enumerate(adjcounts.keys(), start=3))
        word2idxadder = {value: key for key, value in idx2wordadder.items()}
        label2idxadder = {label: i for i, label in enumerate(test, start=1)}
        idx2labeladder = {v: k for k, v in label2idxadder.items()}
        self.idx2word.update(idx2wordadder)
        self.word2idx.update(word2idxadder)
        self.label2idx.update(label2idxadder)
        self.idx2label.update(idx2labeladder)
        return
    
    def convert_text(self):
        '''
        Converts each product desc in the dataset (self.examples) to a list of indices, given by self.word2idx.
        Store this in self.textual_ids; returns nothing.
        '''
        # list of lists; replaces a word with the <UNK> token if it does not exist in the word2idx dictionary.
        self.textual_ids = [[self.word2idx[word] if self.word2idx.get(word) else self.word2idx[UNK] for word in \
                             (Word.upper() for Word in description)] \
                            for (_, description) in self.examples]

        # Appends the <END> token to the end of each description.
        for description in self.textual_ids:
          description.append(self.word2idx[END])
        return
    
    def convert_labels(self):
        '''
        Converts each product category code in the dataset (self.examples) to an index value, given by self.label2idx. 
        Stored in self.label_ids; returns nothing.
        '''
        self.label_ids = [self.label2idx[item[0]] if self.label2idx.get(item[0]) else self.label2idx[UNK] \
                          for item in self.examples] \
                          #for (label, alist) in item]
        return
    
    def get_text(self, idx):
        '''
        Tokenizes product desc.
        Returns the desc at idx as a long tensor (torch.LongTensor) of integers corresponding to the words in the desc.
        May need to PAD.
        '''
        # REVIEW idx
        a = self.textual_ids[idx]
        b = [a[x] if x < len(a) else self.word2idx[PAD] for x in range(0, self.max_len)]
        return torch.LongTensor(b)
    
    def get_label(self, idx):
        '''
        Returns the integer value of the label per idx.
        type = torch.LongTensor.
        '''
        a = self.label_ids[idx]
        return torch.tensor(a, dtype=torch.long)

    def get_labeltext(self, idx):
        '''
        Returns original product desc text per idx.
        type = string.
        '''
        b = self.idx2label[idx]
        return b
    
    def __len__(self):
        '''
        Returns the number of examples (int value) in the dataset
        '''
        return len(self.examples)
    
    def __getitem__(self, idx):
        '''
        Returns the product desc, and label of the example specified by idx.
        '''
        return self.get_text(idx), self.get_label(idx)\


def preprocess_NN(item):
    '''
    #Process each individual item at a time
    '''
    if not item or type(item) != str: return # filter blanks
    #if len(item) < 2: return    
    result = []
    for x in item.split(' '):
      # Trim any needless starting or ending chars from tokens
        if not x or x.isspace(): continue # filter blanks
        else:  
          try:
            remove_beg=True if x[0] in {'(', '"', "'"} else False
            remove_end=True if x[-1] in {'.', ',', ';', ':', '?', '!', '"', "'", ')'} else False
          except:
            print("Error on item: ",item, ord(x))
          if remove_beg and remove_end: result += [x[0], x[1:-1], x[-1]]
          elif remove_beg: result += [x[0], x[1:]]
          elif remove_end: result += [x[:-1], x[-1]]
          else: result += [x]
    return result


def getDataset_NN(table):
    table = pd.DataFrame(table, 
                         header = 0,
                         names = ['Product Description'] 
                         )
    ### TODO figure out the df logistics here, or Series
    test_data = [('TBD', preprocess_NN(x[1])) for x in table.loc[:, ['Product Description']].itertuples(index=False) \
                   if x[0] is not None and type(x[0]) == str]
    return test_data


def read_vocab(path, type):
    # returns dict of {word (str): index (int)}
    vocab = defaultdict()
    print(f'Reading {type}...', end='')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token.strip()] = int(index.strip())
        print(f'COMPLETE: {list(vocab.items())[:3]}')
    return vocab


# NLP-specific properties
PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'
path1 = "app/vocab.pyi"
path2 = "app/labelidx.pyi"
path3 = "app/modelstats.pyi"
vocab_loaded = read_vocab(path1, 'vocab')
labels_loaded = read_vocab(path2, 'labels')
modelstats_loaded = read_vocab(path3, 'stats')
num_classes = len(labels_loaded)
pad_idx = vocab_loaded[PAD] # zero
vocab_size = len(vocab_loaded)
idx2label = {v: k for k, v in labels_loaded.items()}
idx2word = {v: k for k, v in vocab_loaded.items()}
# General model properties
embed_size = 128
out_channels = 64 
filter_heights = [2, 3, 4]
stride = 1
dropout = 0.5 
THRESHOLD=1
MAX_LEN=8

# load model assuming CPU not GPU
model = CNN(vocab_size, embed_size, out_channels, filter_heights, \
    stride, dropout, num_classes, pad_idx)    
PATH = "app/model.ckpt"
model.load_state_dict(torch.load(PATH))
model.eval()


# tokenize input and predict
def get_prediction(input_text, word2idx=vocab_loaded, idx2word=idx2word, \
    label2idx=labels_loaded, idx2label=idx2label, batch_size=1):
    '''
    Input: list [ tuple (string label , list product description tokens) ]
    '''
    # texts = texts.to(device) # ignoring GPU
    # labels = labels.to(device) # ignoring GPU
    # convert to dataloader format
    if input_text is None or type(input_text[0]) is not tuple or len(input_text[0]) != 2:
        raise ValueError('Incorrect input format, correct= [(str(label), [token1, token2]), ...]')
    print('\t Making Ds...', end='')
    Ds = TextDataset(examples=input_text, split='test', threshold=THRESHOLD, max_len=MAX_LEN, \
        word2idx=word2idx, idx2word=idx2word, label2idx=label2idx)
    print('COMPLETE.')
    print('\t Making DataLoader...', end='')
    data_loader = data.DataLoader(Ds, batch_size=batch_size, \
        shuffle=False, num_workers=0, drop_last=False) # no workers for inference
    print('COMPLETE.')
    all_predictions = []
    all_textpredictions = []
    for texts, labels in data_loader:
        output = model(texts) # feed tensor, not labels
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        a = Ds.get_labeltext(pred.item()) # string
        all_textpredictions.append(a)
    print(f'\t Prediction: {all_textpredictions}')
    return all_textpredictions