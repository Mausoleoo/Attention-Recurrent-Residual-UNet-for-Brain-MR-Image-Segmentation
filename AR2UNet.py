import torch
import torch.nn as nn
import torch.nn.functional as F

class Recurrent_block(nn.Module):
    def __init__(self, num_channels, t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(num_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1
    
class R2CL_block(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, t=2):
        super(R2CL_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(*[Recurrent_block(out_channels, t=t) for _ in range(layers)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.relu(x)
        x1 = self.RCNN(x)
        return x + x1
    
class Deconvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconvolution,self).__init__()
        self.Deconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.Deconv(x)
        return x
    
#First parameter is input (x), second parameter is the gate (previous layers)
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels, kernel_size=1),
            nn.BatchNorm2d(gate_channels)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, gate_channels, kernel_size=1),
            nn.BatchNorm2d(gate_channels)
            )
        
        self.psi = nn.Sequential(
            nn.Conv2d(gate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)  
        x1 = self.W_x(x)  
        
        psi = self.relu(g1 + x1)  
        psi = self.psi(psi)  
        alpha = torch.sigmoid(psi)  
        return x * alpha

class AttentionR2UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes = 1):
        super(AttentionR2UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.classes = num_classes
        # Encoder

        self.Encoder1 = R2CL_block(in_channels=input_channels, out_channels=64, t=2)
        self.Encoder2 = R2CL_block(in_channels=64, out_channels=128, t=2)
        self.Encoder3 = R2CL_block(in_channels=128, out_channels=256, t=2)
        self.Encoder4 = R2CL_block(in_channels=256, out_channels=512, t=2)
        self.Encoder5 = R2CL_block(in_channels=512, out_channels=1024, t=2)

        # Attentio Gate
        self.Attention4 = AttentionGate(512, 512)
        self.Attention3 = AttentionGate(256, 256)
        self.Attention2 = AttentionGate(128, 128)
        self.Attention1 = AttentionGate(64, 64)

        # Decoder
        self.Decoder5 = Deconvolution(in_channels=1024, out_channels=512)
        self.Decoder4 = Deconvolution(in_channels=512, out_channels=256)
        self.Decoder3 = Deconvolution(in_channels=256, out_channels=128)
        self.Decoder2 = Deconvolution(in_channels=128, out_channels=64)
        self.Decoder1 = Deconvolution(in_channels=64, out_channels=input_channels)

        self.UPRecurrent4 = R2CL_block(in_channels=1024, out_channels=512, t=2)
        self.UPRecurrent3 = R2CL_block(in_channels=512, out_channels=256, t=2)
        self.UPRecurrent2 = R2CL_block(in_channels=256, out_channels=128, t=2)
        self.UPRecurrent1 = R2CL_block(in_channels=128, out_channels=64, t=2)

        # Output
        self.Conv_1x1 = nn.Conv2d(64, self.classes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Encoder

        x1 = self.Encoder1(x)
        x2 = self.Maxpool(x1)


        x2 = self.Encoder2(x2)
        x3 = self.Maxpool(x2)


        x3 = self.Encoder3(x3)
        x4 = self.Maxpool(x3)


        x4 = self.Encoder4(x4)
        x5 = self.Maxpool(x4)


        x5 = self.Encoder5(x5)
        
        # Decoder
        x5 = self.Decoder5(x5)
        AG = self.Attention4(x5, x4)
        x5 = torch.cat((AG,x5),dim=1)
        x4 = self.UPRecurrent4(x5)
        

        x4 = self.Decoder4(x4)
        AG = self.Attention3(x4, x3)
        x4 = torch.cat((AG,x4),dim=1)
        x3 = self.UPRecurrent3(x4)


        x3 = self.Decoder3(x3)
        AG = self.Attention2(x3, x2)
        x3 = torch.cat((AG,x3),dim=1)
        x2 = self.UPRecurrent2(x3)


        x2 = self.Decoder2(x2)
        AG = self.Attention1(x2, x1)
        x2 = torch.cat((AG,x2),dim=1)
        x1 = self.UPRecurrent1(x2)


        # Output
        x1 = self.Conv_1x1(x1)
        #x1 = self.relu(x1)
        return x1

