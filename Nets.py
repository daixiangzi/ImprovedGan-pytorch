'''
build model for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as Weight_norm
import numpy as np
import plotting
class _G(nn.Module):
	def __init__(self): #inputs (batch,c,w,h)
		super(_G,self).__init__()
		self.main = nn.Sequential(
		nn.ConvTranspose2d(100,512,4,1,0,bias=False),#4*4
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),#8*8
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),#16*16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
		nn.ConvTranspose2d(128,3, 4, 2, 1, bias=False),#32*32
		nn.Tanh()
		)
	def forward(self,inputs):
		return self.main(inputs)
class _D(nn.Module):
        def __init__(self,num_classes):
                super(_D,self).__init__()
                self.main = nn.Sequential(
                        nn.Dropout2d(p=0.2),
                        Weight_norm(nn.Conv2d(3, 96, 3, 1, 1, bias=False),name='weight'), #32*32
                        nn.LeakyReLU(0.2),
                       Weight_norm(nn.Conv2d(96, 96, 3, 1, 1, bias=False),name='weight'), #32*32
                        nn.LeakyReLU(0.2),

                        Weight_norm(nn.Conv2d(96, 96, 3, 2, 1, bias=False),name='weight'), #16*16
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(p=0.5),

                        Weight_norm(nn.Conv2d(96, 192, 3, 1, 1, bias=False),name='weight'), #16*16
                        nn.LeakyReLU(0.2),

                        Weight_norm(nn.Conv2d(192, 192, 3, 1, 1, bias=False),name='weight'), #16*16
                        nn.LeakyReLU(0.2),

                        Weight_norm(nn.Conv2d(192, 192, 3, 2, 1, bias=False),name='weight'), #8*8
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(p=0.5),

                        Weight_norm(nn.Conv2d(192, 192, 3, 1, 0, bias=False),name='weight'), #6*6
                        nn.LeakyReLU(0.2),

                        Weight_norm(nn.Conv2d(192, 192, 1, 1, 0, bias=False),name='weight'), #6*6
                        nn.LeakyReLU(0.2),
                        Weight_norm(nn.Conv2d(192, 192, 1, 1, 0, bias=False),name='weight'), #6*6
                        nn.LeakyReLU(0.2),
                        )
                self.main2 = nn.Sequential(
                Weight_norm(nn.Linear(192,num_classes),name='weight')
                        )
        def forward(self,inputs,feature=False):
                output = self.main(inputs)# 192*6*6
                output = F.adaptive_avg_pool2d(output,[1,1])#192*1*1
                if feature:
                    return output
                output = output.view(-1,192)#1*192
                features = self.main2(output)#1*10
                return torch.squeeze(features) #10
class Train(object):
    def __init__(self,G,D,G_optim,D_optim):
        self.G = G
        self.D = D
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.noise = torch.randn(100, 100, 1, 1)
    def log_sum_exp(self,x, axis = 1):
        m = torch.max(x, dim = 1)[0]
        return torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))+m
    def train_batch_disc(self,x,y,x_unlabel):
        noise = torch.randn(y.shape[0], 100, 1, 1)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()
            x_unlabel = x_unlabel.cuda()
            y = y.cuda()

        self.D.train()
        self.G.train()

        lab = self.D(x)
        unlab = self.D(x_unlabel)
        gen = self.G(noise)
        gen_output = self.D(gen)
        #loss
        loss_lab = torch.mean(torch.mean(self.log_sum_exp(lab)))-torch.mean(torch.gather(lab, 1, y.unsqueeze(1)))
        loss_unlab = 0.5*(torch.mean(F.softplus(self.log_sum_exp(unlab)))-torch.mean(self.log_sum_exp(unlab))+torch.mean(F.softplus(self.log_sum_exp(gen_output))))
        total_loss = loss_lab + loss_unlab
        # acc
        acc = torch.mean((lab.max(1)[1] == y).float())
        self.D_optim.zero_grad()
        total_loss.backward()
        self.D_optim.step()
        return loss_lab.item(),loss_unlab.item(),acc.item()

    def train_batch_gen(self,x_unlabel):
        noise = torch.randn(x_unlabel.shape[0], 100, 1, 1)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x_unlabel = x_unlabel.cuda()

        self.D.train()
        self.G.train()

        gen_data = self.G(noise)
        output_unl = self.D(x_unlabel,feature=True)
        output_gen = self.D(gen_data,feature=True)
        
        m1 = torch.mean(output_unl,dim=0)
        m2 = torch.mean(output_gen,dim=0)
        loss_gen = torch.mean(torch.abs(m1-m2))
        self.G_optim.zero_grad()
        self.D_optim.zero_grad()
        loss_gen.backward()
        self.G_optim.step()
        return loss_gen.item()
        
    def update_learning_rate(self,lr):       
        for param_group in self.G_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.D_optim.param_groups:
            param_group['lr'] = lr
    def test(self,x,y):
        self.D.eval()
        self.G.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                x,y = x.cuda(),y.cuda()
            output = self.D(x)
            acc = torch.mean((output.max(1)[1] == y).float())
        return acc.item()
    def save_png(self,save_dir,epoch):
        if torch.cuda.is_available():
            noise = self.noise.cuda()
        with torch.no_grad():
            gen_data = self.G(noise)
        gen_data = gen_data.cpu().detach().numpy()
        img_bhwc = np.transpose(gen_data, (0, 2, 3, 1))
        img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples '+str(epoch)+" epochs")
        plotting.plt.savefig(save_dir+"cifar_sample_feature_match_"+str(epoch)+".png")

