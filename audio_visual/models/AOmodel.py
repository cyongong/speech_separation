
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

'''

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class AVNet(nn.Module):
     def __init__(self,people_num=2):
        super(AVNet, self).__init__()
        model_input = torch.randn(1,2,298,257)
        print('0:', model_input.shape)

        conv1 = nn.Conv2d(2,96, kernel_size=(1, 7), stride=(1, 1), padding='same', dilation=(1, 1))
        conv1to14_bn = nn.BatchNorm2d(96) #this part has diff param s a eps momentum, check! 1 to 14 layers have same output ch size! 
        conv_at = nn.ReLU # all layers have a same activation fc.
        #print('1:', conv1.shape)

        conv2 = nn.Conv2d(2,96, kernel_size=(7,1), stride=(1, 1), padding='same', dilation=(1, 1))
        conv3 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(1, 1))
        conv4 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(2, 1))
        conv5 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(4, 1))
        conv6 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(8, 1))
        conv7 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(16, 1))
        conv8 = nn.Conv2d(96,96, kernel_size=(7,1), stride=(1, 1), padding='same', dilation=(32, 1))
        conv9 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(1, 1))
        conv10 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(2,2))
        conv11 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(4,4))
        conv12 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(8,8))
        conv13 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(16, 16))
        conv14 = nn.Conv2d(96,96, kernel_size=(5,5), stride=(1, 1), padding='same', dilation=(32, 32))
        conv15 = nn.Conv2d(96,8, kernel_size=(1,1), stride=(1, 1), padding='same', dilation=(1, 1))
        conv15_bn = nn.BatchNorm2d(8)
        
        Flt = nn.Flatten(0,3) #flatten 4 dim to 1
        td = TimeDistributed()
        rnn =nn.LSTM(8,400,1,batch_first=True)
        ds1 = nn.Linear(400,600)  #it is diff from keras bof seed dropout.
        ds2 = nn.Linear(600,600)
        ds3 = nn.Linear(600,600)
        cp_mask = nn.Linear(600, 257*2*2) #diff from keras bof complex mask kernel initialization
        
        #bidirectional lstm part is needed!!!!

     def forward(self,x):

         output = conv1(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv2(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv3(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv4(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv5(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv6(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv7(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv8(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv9(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv10(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv11(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv12(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv13(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv14(model_input)
         output = conv1to14_bn(output)
         output = conv_at(output)

         output = conv15(model_input)
         output = conv15_bn(output)
         output = conv_at(output)

         output = td(Flt(output))

         h0 = torch.randn(1,298,400)  #input =(batch, seq_len, feature)
         c0 = torch.randn(1,298,400)  #(1, batch, output_feature)

         output, (hn,cn) = rnn(output,(h0,c0))

         fc1 = conv_at(ds1(output))
         fc2 = conv_at(ds2(fc1))
         fc3 = conv_at(ds3(fc2))

         complex_mask = cp_mask(fc3)
         complex_mask_out = torch.reshape(complex_mask, (298, 257, 2, people_num) )


        
       
         return complex_mask_out


   