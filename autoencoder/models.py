import math

from collections import OrderedDict

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score, roc_curve, auc

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
            ('activation', nn.SELU())
        ]))
        self.decoder = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hid_dim, inp_dim))
            # ('activation', nn.SELU())
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
            unit.append(('activation', nn.SELU()))
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
        decoder_units[-1].add_module('activation', nn.CELU())
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
        # self.hidden[-1].add_module('activation', nn.Softmax())


    def forward(self, inp):
        return self.hidden(inp)


class conv2d_net(nn.Module):
    ''' 2d conv over spatial domain '''

    def __init__(self, input_dim, input_w, input_h, output_dim):
        super(conv2d_net, self).__init__()
        self.inp_w = input_w
        self.inp_h = input_h
        self.out_dim = output_dim
        kernel_size = 5
        kernel_num = (20,)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, kernel_num[0], 
                        kernel_size=kernel_size, 
                        stride=1),
            nn.ReLU(),
            # nn.Conv2d(kernel_num[0], kernel_num[1], 
            #             kernel_size=kernel_size, 
            #             stride=1, 
            #             padding=int((kernel_size-1)*0.5)),
            # nn.ReLU(),
        )
        fc_inp_dim = 1 * kernel_num[0]
        self.fc_inp_dim = fc_inp_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_inp_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU()
        )
        self.conv_alex = nn.Sequential(
            nn.Conv2d(kernel_num[0], 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((output_dim, None, None))
        )
        self.out_layer = nn.Linear(10, output_dim)

    
    def forward(self, inp):
        inp = torch.reshape(inp, (1, -1, self.inp_h, self.inp_w))
        conv_out = self.conv(inp)
        _, conv_c, _, _ = conv_out.shape
        conv_out = torch.reshape(conv_out, (1,)+conv_out.shape)
        conv_out = nn.functional.interpolate(conv_out, size=(conv_c, self.inp_h, self.inp_w))
        # use normal fc
        conv_out = torch.reshape(conv_out, (self.fc_inp_dim, -1))
        conv_out = conv_out.T
        fc_out = self.fc(conv_out)
        out = self.out_layer(fc_out)
        # use conv_alex
        # conv_out = torch.reshape(conv_out, conv_out.shape[1:])
        # conv_alex_out = self.conv_alex(conv_out)
        # out = torch.reshape(conv_alex_out, (self.out_dim, -1))
        # out = out.T
        return out


class fconv2d_net(nn.Module):
    ''' 2d fully conv over spatial domain '''

    def __init__(self, input_dim, input_w, input_h, output_dim):
        super(fconv2d_net, self).__init__()
        self.inp_w = input_w
        self.inp_h = input_h
        self.out_dim = output_dim
        kernel_size = 5
        kernel_num = (20,)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, kernel_num[0], 
                        kernel_size=kernel_size, 
                        stride=1),
            nn.ReLU(),
        )
        self.transp_conv = nn.ConvTranspose2d(kernel_num[0], input_dim,
                                                kernel_size=kernel_size,
                                                stride=1)
        self.conv_alex = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((output_dim, None, None))
        )

    
    def forward(self, inp):
        inp = torch.reshape(inp, (1, -1, self.inp_h, self.inp_w))
        conv_out = self.conv(inp)
        transp_conv_out = self.transp_conv(conv_out)
        # use conv_alex
        conv_alex_out = self.conv_alex(transp_conv_out)
        out = torch.reshape(conv_alex_out, (self.out_dim, -1))
        out = out.T
        return out


class conv3d_net(nn.Module):
    ''' 3d conv over spectral-spatial domain'''

    def __init__(self, input_dim, input_w, input_h, output_dim):
        super(conv3d_net, self).__init__()
        self.inp_w = input_w
        self.inp_h = input_h
        self.inp_dim = input_dim
        kernel_size = 7
        kernel_num = 20

        self.conv = nn.Sequential(
            nn.Conv3d(1, kernel_num, kernel_size=kernel_size, stride=1),
            nn.ReLU()
        )
        fc_inp_dim = kernel_num * input_dim
        self.fc_inp_dim = fc_inp_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_inp_dim, 100),
            nn.ReLU()
        )
        self.out_layer = nn.Linear(100, output_dim)

    def forward(self, inp):
        inp = torch.reshape(inp, (1, 1, -1, self.inp_h, self.inp_w))
        conv_out = self.conv(inp)
        conv_out = nn.functional.interpolate(conv_out, size=(self.inp_dim, self.inp_h, self.inp_w))
        conv_out = torch.reshape(conv_out, (self.fc_inp_dim, -1))
        conv_out = conv_out.T
        fc_out = self.fc(conv_out)
        out = self.out_layer(fc_out)
        return out


class fconv3d_net(nn.Module):
    ''' 3d fully conv over spectral-spatial domain'''

    def __init__(self, input_dim, input_w, input_h, output_dim):
        super(fconv3d_net, self).__init__()
        self.inp_w = input_w
        self.inp_h = input_h
        self.out_dim = output_dim
        kernel_size = 7
        kernel_num = 20

        self.conv = nn.Sequential(
            nn.Conv3d(1, kernel_num, kernel_size=kernel_size, stride=1),
            nn.ReLU()
        )
        self.transp_conv = nn.ConvTranspose3d(kernel_num, 1, kernel_size=kernel_size, stride=1)
        self.conv_alex = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv3d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((1, None, None))
        )
        self.out_layer = nn.AdaptiveMaxPool3d((output_dim, None, None))


    def forward(self, inp):
        inp = torch.reshape(inp, (1, 1, -1, self.inp_h, self.inp_w))
        conv_out = self.conv(inp)
        transp_conv_out = self.transp_conv(conv_out)
        conv_alex_out = self.conv_alex(transp_conv_out)
        conv_alex_out = torch.reshape(conv_alex_out, (1, 1, -1, self.inp_h, self.inp_w))
        out = self.out_layer(conv_alex_out)
        out = torch.reshape(out, (self.out_dim, -1))
        out = out.T
        return out


class conv1d_net(nn.Module):
    ''' 1d conv over spectral domain '''

    def __init__(self, input_dim, output_dim):
        super(conv1d_net, self).__init__()
        self.inp_dim = input_dim
        kernel_size = 7
        kernel_num = 20
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, kernel_num, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        fc_inp_dim = int((input_dim - kernel_size) * 0.5 * kernel_num)
        self.fc_inp_dim = fc_inp_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_inp_dim, 100),
            nn.ReLU()
        )
        self.out_layer = nn.Linear(100, output_dim)


    def forward(self, inp):
        inp = torch.reshape(inp, (-1, 1, self.inp_dim))
        conv_out = self.conv(inp)
        conv_out = torch.reshape(conv_out, (-1, self.fc_inp_dim))
        fc_out = self.fc(conv_out)
        out = self.out_layer(fc_out)
        return out


class conv_on_patch(nn.Module):
    ''' 1x1 conv on patch '''

    def __init__(self, input_dim, output_dim):
        super(conv_on_patch, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        kernel_size = 1
        kernel_num = (128, 64)

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, kernel_num[0], kernel_size=kernel_size),
            nn.BatchNorm2d(kernel_num[0]),
            nn.ReLU(),
            nn.Conv2d(kernel_num[0], kernel_num[1], kernel_size=kernel_size),
            nn.BatchNorm2d(kernel_num[1]),
            nn.ReLU(),
            nn.Conv2d(kernel_num[1], output_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, inp):
        inp = torch.transpose(inp, 1, 2)
        _, _, inp_l = inp.shape
        patch_size = int(math.sqrt(inp_l))
        inp = torch.reshape(inp, (-1, self.inp_dim, patch_size, patch_size))
        conv_out = self.conv(inp)
        out = torch.reshape(conv_out, (-1, self.out_dim))
        return out


class conv_hybrid(nn.Module):
    ''' 2d-convnet + inception module '''

    def __init__(self, input_dim, output_dim):
        super(conv_hybrid, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        
        incep_output_dim = 128
        self.inception1 = nn.Sequential(
            nn.Conv2d(input_dim, incep_output_dim, kernel_size=1),
            nn.MaxPool2d(3)
        )
        self.inception2 = nn.Conv2d(input_dim, incep_output_dim, kernel_size=3)
        self.conv_cat = nn.Sequential(
            nn.ReLU(),
            nn.LocalResponseNorm(3, k=2),
            nn.Conv2d(incep_output_dim*2, 128, kernel_size=1),
            nn.ReLU(),
            nn.LocalResponseNorm(3, k=2)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
        )
        self.activat = nn.ReLU()
        self.conv_alex = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((output_dim, None, None))
        )

    def forward(self, inp):
        inp = torch.transpose(inp, 1, 2)
        _, _, inp_l = inp.shape
        patch_size = int(math.sqrt(inp_l))
        inp = torch.reshape(inp, (-1, self.inp_dim, patch_size, patch_size))
        incep1_out = self.inception1(inp)
        incep2_out = self.inception2(inp)
        # depth concat
        incep_out = torch.cat((incep1_out, incep2_out), dim=1)
        conv_cat_out = self.conv_cat(incep_out)
        # residual learning
        resid_out = self.residual(conv_cat_out)
        # sum
        sum_out = torch.add(resid_out, conv_cat_out)
        sum_out = self.activat(sum_out)
        # residual learning
        resid_out = self.residual(sum_out)
        # sum
        sum_out = torch.add(resid_out, sum_out)
        # conv as alexnet
        conv_alex_out = self.conv_alex(sum_out)
        out = torch.reshape(conv_alex_out, (-1, self.out_dim))
        return out


class conv_incep(nn.Module):
    ''' inception module '''

    def __init__(self, input_dim, output_dim):
        super(conv_incep, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim

        incep_output_dim = 128
        self.inception1 = nn.Sequential(
            nn.Conv2d(input_dim, incep_output_dim, kernel_size=1),
            nn.MaxPool2d(3)
        )
        self.inception2 = nn.Conv2d(input_dim, incep_output_dim, kernel_size=3)
        self.conv_cat = nn.Sequential(
            nn.ReLU(),
            nn.LocalResponseNorm(3, k=2),
            nn.Conv2d(incep_output_dim*2, 128, kernel_size=1),
            nn.ReLU(),
            nn.LocalResponseNorm(3, k=2)
        )
        self.conv_alex = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((output_dim, None, None))
        )


    def forward(self, inp):
        inp = torch.transpose(inp, 1, 2)
        _, _, inp_l = inp.shape
        patch_size = int(math.sqrt(inp_l))
        inp = torch.reshape(inp, (-1, self.inp_dim, patch_size, patch_size))
        incep1_out = self.inception1(inp)
        incep2_out = self.inception2(inp)
        # depth concat
        incep_out = torch.cat((incep1_out, incep2_out), dim=1)
        conv_cat_out = self.conv_cat(incep_out)
        # conv as alexnet
        conv_alex_out = self.conv_alex(conv_cat_out)
        out = torch.reshape(conv_alex_out, (-1, self.out_dim))
        return out


class conv_resid(nn.Module):
    ''' residual learning module '''

    def __init__(self, input_dim, output_dim):
        super(conv_resid, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
        )
        self.conv_alex = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.AdaptiveMaxPool3d((output_dim, None, None))
        )
        

    def forward(self, inp):
        inp = torch.transpose(inp, 1, 2)
        _, _, inp_l = inp.shape
        patch_size = int(math.sqrt(inp_l))
        inp = torch.reshape(inp, (-1, self.inp_dim, patch_size, patch_size))
        conv1_out = self.conv1(inp)
        # residual learning
        resid_out = self.residual(conv1_out)
        # sum
        sum_out = torch.add(resid_out, conv1_out)
        # conv as alexnet
        conv_alex_out = self.conv_alex(sum_out)
        out = torch.reshape(conv_alex_out, (-1, self.out_dim))
        return out



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


def get_roc(dataset: Dataset, model):
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

    fpr, tpr, _ = roc_curve(grd_truth.numpy(), pred_res.numpy(), pos_label=2)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def predict_class(dataset: Dataset, model):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    pred_res = []
    with torch.no_grad():
        for data in dataloader:
            inp = data[1]
            if hasattr(model, 'encoder'):
                output = model.encoder(inp)
            else:
                output = model(inp)
            _, prediction = torch.max(output.data, 1)

            loc = data[0]
            if len(pred_res) <= 0:
                pred_res = prediction
                continue
            pred_res = torch.cat((pred_res, prediction))
            
    return pred_res


def get_model_output(dataset: Dataset, model):
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    output_res = []
    with torch.no_grad():
        for data in dataloader:
            inp = data[1]
            if hasattr(model, 'encoder'):
                output = model.encoder(inp)
            else:
                output = model(inp)
            if len(output_res) <= 0:
                output_res = output
                continue
            output_res = torch.cat((output_res, output))
            
    return output_res