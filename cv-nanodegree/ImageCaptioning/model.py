import torch
import torch.nn as nn
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
        
        self.hidden_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_dim, num_layers, batch_first=True)
        self.drop_out = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(self.hidden_dim, vocab_size)
        
        self.hidden = self.init_hidden()
        
        # initialize the weights
        self.init_hidden()
        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
        
    def forward(self, features, captions):
        cap_embedding = self.word_embeddings(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(embeddings)
        
        outputs = self.fc(lstm_out)
        
        #tag_outputs = self.fc(lstm_out.contiguous().view(len(captions), -1))
        #outputs = F.log_softmax(tag_outputs, dim=1)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass