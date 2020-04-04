from collections import OrderedDict

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score

import torch
from torch import nn
from torch import optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader, TensorDataset



class simple_ae(nn.Module):
    def __init__(self, inp_size):
        super(simple_ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inp_size, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True), nn.Linear(8, 4), nn.ReLU(True), nn.Linear(4, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(True),
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True), nn.Linear(16, inp_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_simple_ae(dataset, inp_dim, learning_rate, num_epoch, save_path=None):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    model = simple_ae(inp_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # shuffle dataset
    # for train, test in ShuffleSplit(n_splits=2, test_size=0.1).split(data_idx):
    #     print(train.shape)

    for epoch in range(num_epoch):
        for data in dataloader:
            inp = data[1]
            
            output = model(inp)
            loss = criterion(output, inp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log
        print('epoch [{}/{}], loss:{:.4f}' 
                .format(epoch + 1, num_epoch, loss.data.item()))

    if save_path:
        torch.save(model.state_dict(), save_path)


class dae(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super(dae, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(inp_dim, hid_dim)), 
            ('activation', nn.ReLU())
        ]))
        self.decoder = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hid_dim, inp_dim)), 
            ('activation', nn.Softplus())
        ]))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def copy_weights(self, encoder: nn.Linear, decoder: nn.Linear):
        encoder.weight.data.copy_(self.encoder.linear.weight)
        encoder.bias.data.copy_(self.encoder.linear.bias)
        decoder.weight.data.copy_(self.decoder.linear.weight)
        decoder.bias.data.copy_(self.decoder.linear.bias)


def build_units(dimensions):
    ''' unit: (Linear + ReLU) '''

    units = []
    for i in range(len(dimensions)-1):
        inp_dim = dimensions[i]
        outp_dim = dimensions[i+1]
        unit = [('linear', nn.Linear(inp_dim, outp_dim))]
        if outp_dim != dimensions[-1]:
            unit.append(('activation', nn.ReLU()))
        units.append(nn.Sequential(OrderedDict(unit)))

    return units


class sdae(nn.Module):
    ''' layer-wise training '''

    def __init__(self, dimensions):
        super(sdae, self).__init__()
        self.dimensions = dimensions
        encoder_units = build_units(dimensions)
        self.encoder = nn.Sequential(*encoder_units)
        decoder_units = build_units(list(reversed(dimensions)))
        decoder_units[-1].add_module('activation', nn.Softplus())
        # unit = [('activation', nn.Softplus())]
        # decoder_units.append(nn.Sequential(OrderedDict(unit)))
        self.decoder = nn.Sequential(*decoder_units)

    def forward(self, inp):
        return self.decoder(self.encoder(inp))

    def get_stack(self, index):
        return self.encoder[index].linear, self.decoder[-(index+1)].linear



def train_ae(dataset: Dataset, autoencoder: nn.Module, num_epoch):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9)
    autoencoder.train()

    for epoch in range(num_epoch):
        for data in dataloader:
            if type(data) == list:
                inp = data[1]
            else:
                inp = data
            output = autoencoder(inp)
            loss = criterion(output, inp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log
        print('epoch [{}/{}], loss:{:.4f}' 
                .format(epoch + 1, num_epoch, loss.data.item()))
        

        
def pretrain(dataset: Dataset, autoencoder: sdae, num_epoch):
    current_dataset = dataset
    num_subae = len(autoencoder.dimensions)-1
    for i in range(num_subae):
        inp_dim = autoencoder.dimensions[i]
        hid_dim = autoencoder.dimensions[i+1]
        encoder, decoder = autoencoder.get_stack(i)

        sub_ae = dae(inp_dim=inp_dim, hid_dim=hid_dim)
        train_ae(current_dataset, sub_ae, num_epoch)

        sub_ae.copy_weights(encoder, decoder)
        if i < num_subae-1:
            current_dataset = encode(current_dataset, sub_ae)

    # train_ae(dataset, autoencoder, num_epoch)


def encode(dataset: Dataset, autoencoder: nn.Module):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    coded_res = []
    with torch.no_grad():
        for data in dataloader:
            if type(data) == list:
                inp = data[1]
            else:
                inp = data
            if hasattr(autoencoder, 'encoder'):
                output = autoencoder.encoder(inp)
            else:
                output = autoencoder(inp)
            if len(coded_res) <= 0:
                coded_res = output
                continue
            coded_res = torch.cat((coded_res, output))

    return coded_res



def fine_tune(dataset: Dataset, autoencoder: sdae, num_epoch, 
                validation=None, train_encoder_more=True):
    train_ae(dataset, autoencoder, num_epoch)
    if not train_encoder_more:
        return autoencoder
    model = sdae_lr(autoencoder)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    for epoch in range(num_epoch):
        for data in dataloader:
            inp = data[1]
            truth = data[2]
            output = model(inp)
            loss = criterion(output, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log
        if validation is not None:
            acc = cal_accuracy(validation, model)
            print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' 
                .format(epoch + 1, num_epoch, loss.data.item(), acc))
        else:
            print('epoch [{}/{}], loss:{:.4f}' 
                .format(epoch + 1, num_epoch, loss.data.item()))

    return model


class sdae_lr(nn.Module):
    ''' sdae + logistic regression '''

    def __init__(self, autoencoder: sdae):
        super(sdae_lr, self).__init__()
        self.hidden = autoencoder.encoder
        # self.hidden[-1].add_module('activation', nn.Sigmoid())


    def forward(self, inp):
        return self.hidden(inp)


def cal_accuracy(dataset: Dataset, model):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    correct = 0
    pred_res = []
    grd_truth = []
    with torch.no_grad():
        for data in dataloader:
            inp = data[1]
            truth = data[2]
            if hasattr(model, 'encoder'):
                output = model.encoder(inp)
            else:
                output = model(inp)
            _, prediction = torch.max(output.data, 1)
            correct += (prediction == truth).sum()
            #
            if len(pred_res) <= 0:
                pred_res = prediction
                grd_truth = truth
                continue
            pred_res = torch.cat((pred_res, prediction))
            grd_truth = torch.cat((grd_truth, truth))

    precision = precision_score(grd_truth.numpy(), pred_res.numpy(), average='micro')

    self_cal_acc = correct.item()/len(pred_res)

    return precision

