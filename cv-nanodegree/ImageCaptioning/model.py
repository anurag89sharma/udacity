import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.dense_bn = nn.BatchNorm1d(embed_size)
        self.fc_drop = nn.Dropout(p=0.2)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc_drop(self.dense_bn(self.embed(features)))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob=0.2):
        super(DecoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_size
        #self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)
        #self.hidden = self.init_hidden(batch_size)
        #self.hidden = None
        
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state; which will be
           none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim), 
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        
    def forward(self, features, captions):
        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size)
        cap_embedding = self.word_embeddings(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        #lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out, self.hidden = self.lstm(embeddings)
        print("length - {} shape - {}".format(len(self.hidden), self.hidden[0].shape))
        
        outputs = self.fc(lstm_out)
        #outputs = F.log_softmax(tag_outputs, dim=1)
        
        return outputs

    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (num_layers, 1, embed_size)
        #hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        
        while True:
            lstm_out, hidden = self.lstm(inputs) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.fc(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
            if (max_indice == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        return output