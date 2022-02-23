import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import random
import os
import pandas
from tqdm.notebook import tqdm
import csv


'''
STEP ONE: IMPORT DATA
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    print('Using device:', device)


def getDataset_NN():
    """
    Returns 2 dataframes of Code, Name, PRODHA, dataset, where PRODHA is '?' in the test set
    """
    df = []
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "../data/Test6_REF.csv"
    abs_file_path = os.path.join(script_dir, rel_path)

    path = abs_file_path

    filedata = pandas.read_csv(path, 
                         header = 0, 
                         names = ['Code', 'Name', 'PRODHA', 'dataset'] 
                         )

    df.append(filedata[filedata['dataset'] == 'train']) #df[0]
    df.append(filedata[filedata['dataset'] == 'test']) #df[1]

    return df[0], df[1]


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


if __name__=='__main__':
    train_data1, test_data1 = getDataset_NN()
    train_data1 = [(x[0], preprocess_NN(x[1])) for x in train_data1.loc[:, ['PRODHA','Name']].itertuples(index=False) \
                   if x[1] is not None and x[0] is not None and type(x[1]) == str]
    test_data1 = [(x[0], preprocess_NN(x[1])) for x in test_data1.loc[:, ['PRODHA','Name']].itertuples(index=False) \
                   if x[1] is not None and x[0] is not None and type(x[1]) == str]
    #train_data, test_data = train_data[0:10000] + train_data[12500:12500+10000], train_data[10000:12500] + train_data[12500+10000:], 

    # print(train_data1[1:10])  
    print('Num. Train Examples:', len(train_data1))
    print('Num. Test Examples:', len(test_data1))

    print("\nSAMPLE TRAIN DATA:")
    for x in random.sample(train_data1, 5):
        print('Sample text:', x[1])
        print('Sample label:', x[0], '\n')
    
    print("\nSAMPLE TEST DATA:")
    for x in random.sample(test_data1, 5):
        print('Sample text:', x[1])
        print('Sample label:', x[0], '\n')


'''
STEP TWO: PYTORCH DATA CLASS
'''


PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'
from collections import Counter

class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None, label2idx=None):

        self.examples = examples
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

        # Appends the <END> token to the end of each review.
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
        Returns original product hierarchy text per idx.
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


if __name__=='__main__':
    # For checking only. Ignored downstream.
    Ds = TextDataset(train_data1, split='train', threshold=1, max_len=8)
    print('Vocab size:', Ds.vocab_size)
    print('Label set:', Ds.label_size)

    text, label = Ds[random.randint(0, len(Ds))]
    print('Example text:', text)
    print('Example label:', label)

    Ds2 = TextDataset(test_data1, split='test', threshold=1, max_len=8,\
         word2idx=Ds.word2idx, idx2word=Ds.idx2word, label2idx=Ds.label2idx)
    print('Vocab size:', Ds2.vocab_size)
    print('Label set:', Ds2.label_size)
    
    text, label = Ds2[random.randint(0, len(Ds2))]
    print('Example text:', text)
    print('Example label:', label)


'''
STEP 3: CREATE DATALOADER, TRAIN AND SAVE THE CNN
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


if __name__=='__main__':
    THRESHOLD = 1
    MAX_LEN = 8
    BATCH_SIZE = 32 

    train_Ds = TextDataset(train_data1, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_Ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_Ds = TextDataset(test_data1, 'test', THRESHOLD, MAX_LEN, train_Ds.idx2word, train_Ds.word2idx, train_Ds.label2idx)
    test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    print('COMPLETED DATALOADER')


def train_model(model, num_epochs, data_loader, optimizer, criterion):
    print('Training Model...')
    model.train()
    for epoch in tqdm(range(num_epochs), miniters=1, leave=True):
        epoch_loss = 0
        epoch_acc = 0
        for texts, labels in data_loader:
            texts = texts.to(device) # shape: [batch_size, MAX_LEN]
            labels = labels.to(device) # shape: [batch_size]

            optimizer.zero_grad() # zero the gradients

            output = model(texts)
            acc = accuracy(output, labels)
            
            loss = criterion(output, labels) # define loss function
            loss.backward() # compute gradients
            optimizer.step() # update weights

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, epoch_loss/len(data_loader), 100*epoch_acc/len(data_loader)))
    print('Model Trained!\n')
    
    torch.save(model.state_dict(), 'model.ckpt')


def count_parameters(model):
    """
    Count number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, labels):
    """
    Returns accuracy per batch
    output: Tensor [batch_size, n_classes]
    labels: LongTensor [batch_size]
    """
    preds = output.argmax(dim=1) # find predicted class
    correct = (preds == labels).sum().float() # convert into float for division 
    acc = correct / len(labels)
    return acc


if __name__=='__main__':
    cnn_model = CNN(vocab_size = train_Ds.vocab_size, 
                embed_size = 128, 
                out_channels = 64, 
                filter_heights = [2, 3, 4], 
                stride = 1, 
                dropout = 0.5, 
                num_classes = train_Ds.label_size, 
                pad_idx = train_Ds.word2idx[PAD])

    # Put model on the device (cuda or cpu)
    cnn_model = cnn_model.to(device)
    
    print('The model has {:,d} trainable parameters'.format(count_parameters(cnn_model)))


if __name__=='__main__':    
    LEARNING_RATE = 5e-4 

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Define the optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)


if __name__=='__main__':    
    N_EPOCHS = 25 
    
    # train model for N_EPOCHS epochs
    # train_model(cnn_model, N_EPOCHS, train_loader, optimizer, criterion)


'''
STEP 4: EVALUATE
'''


def evaluate(model, data_loader, criterion, dataset):
    print('Evaluating performance on the test dataset...')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_textpredictions = []
    print("\nSOME PREDICTIONS FROM THE MODEL:")
    for texts, labels in tqdm(data_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        
        output = model(texts)
        acc = accuracy(output, labels)
        pred = output.argmax(dim=1)
        
        all_predictions.append(pred)
        a = dataset.get_labeltext(pred.item()) # string
        all_textpredictions.append(a)
        
        loss = criterion(output, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if random.random() < 0.01:
            print("Input: "+' '.join([data_loader.dataset.idx2word[idx] for idx in texts[0].tolist() if idx not in {data_loader.dataset.word2idx[PAD], data_loader.dataset.word2idx[END]}]))
            print("Prediction:", pred.item(), '\tCorrect Output:', labels.item(), '\n')

    full_acc = 100*epoch_acc/len(data_loader)
    full_loss = epoch_loss/len(data_loader)
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
   

    return predictions, full_acc, full_loss, all_textpredictions


if __name__=='__main__':
    _ , _ , _ , output = evaluate(cnn_model, test_loader, criterion, test_Ds) # Compute test data accuracy


'''
STEP 5: EXPORT
'''


def save_vocab(vocab_dict, path):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab_dict.items():
            f.write(f'{index}\t{token}\n')
    

if __name__=='__main__':
    # save vocab word2idx for inference, vocab length, PAD index
    path = "app/vocab.pyi"
    vocab_dict = train_Ds.word2idx
    save_vocab(vocab_dict, path)
    print(f'Vocab saved to {path}.' + '\n' )

    # save label index label2idx for inference, num labels
    path2 = "app/labelidx.pyi"
    label_dict = train_Ds.label2idx
    save_vocab(label_dict, path2)
    print(f'Labels saved to {path2}.' + '\n')

    # save model statistics
    path3 = "app/modelstats.pyi"
    stat_dict = {'rows trained on': len(train_Ds.examples), \
        'label count': train_Ds.label_size, \
        'vocab size': train_Ds.vocab_size}
    save_vocab(stat_dict, path3)
    print(f'Model stats saved to {path3}.' + '\n' + '\n')


# save predictions if unlabeled examples loaded as test_Ds
if __name__=='__main__':
    _ , _ , _ , output = evaluate(cnn_model, test_loader, criterion, test_Ds) # Compute test data accuracy
    with open('CNNoutput.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(output)