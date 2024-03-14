import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from einops import rearrange
from torch.optim.lr_scheduler import StepLR
import numpy as np


BATCH_SIZE=512 # 批次大小
EPOCHS=30 # 总共训练批次
DEVICE='cuda:0'

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # layer norm
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # layer norm
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # layer norm
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # batch norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        
        out = self.bn(out)
        out = self.activation(out)
        return out
    
class Resnet(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = conv_block(in_dim, out_dim, kernel_size, stride, padding)
        self.conv2 = conv_block(out_dim, out_dim, kernel_size, stride, padding)
        self.con3 = nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding)
        # # batch norm
        self.norm = nn.BatchNorm2d(out_dim)
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.con3(out)
        
        out = self.norm(out)
        return out
    
class res_block(nn.Module):
    def __init__(self, in_dim, out_dim,  kernel_size=3,stride=1, padding=1):
        super().__init__()
        self.conv1 = conv_block(in_dim, out_dim, kernel_size, stride, padding)
        self.conv2 = Resnet(out_dim, out_dim, kernel_size, stride, padding)
        self.conv3 = conv_block(out_dim, out_dim, kernel_size, stride, padding)
    
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        res = out1 + out2
        out = self.conv3(res)
        return out
    
class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        # conv
        self.conv1 = res_block(in_dim, out_dim)
        self.pool1 = conv_block(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = res_block(out_dim, out_dim*2)
        self.pool2 = conv_block(out_dim*2, out_dim*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = res_block(out_dim*2, out_dim*4)
        self.pool3 = conv_block(out_dim*4, out_dim*4, kernel_size=3, stride=2, padding=1)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        return out
    
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # attention
        self.attn = Transformer(dim=256, depth=6, heads=8, dim_head=64, mlp_dim=256, dropout=0.1)
        # encoder
        self.encoder = Encoder(in_dim=3, out_dim=24)
        
        # layer norm
        self.norm1 = nn.LayerNorm(1536)
        self.norm2 = nn.LayerNorm(256)
        # linear
        # self.fc1 = nn.Linear(480,256)
        self.fc2 = nn.Linear(256, 10)
        self.to_latent = nn.Identity()
        # fcn
        self.fc1 = nn.Conv1d(1536, 256, kernel_size=1, stride=1, padding=0)
        # self.fc2 = nn.Conv1d(256, 10, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x):
        in_size = x.size(0)
        out = self.encoder(x)
        out = out.view(in_size,-1)
        out = self.norm1(out)
        out = out.transpose(0,1)
        out = self.fc1(out)
        out = out.transpose(0,1)
        out = self.norm2(out)
        out = out.unsqueeze(1)
        
        out = self.attn(out)
        out = self.to_latent(out)
        out = out.squeeze(1)
        out = self.fc2(out)
        # print("out shape is ",out.shape)
        out = F.log_softmax(out,dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model = model.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model = model.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
        
if __name__ == "__main__":
    
    # MNIST数据集
    # train_loader = torch.utils.data.DataLoader(
    #         datasets.MNIST('data', train=True, download=True,
    #                     transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.1307,), (0.3081,))
    #                     ])),
    #         batch_size=BATCH_SIZE, shuffle=True)


    # test_loader = torch.utils.data.DataLoader(
    #         datasets.MNIST('data', train=False, transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.1307,), (0.3081,))
    #                     ])),
    #         batch_size=BATCH_SIZE, shuffle=True)
    
    # # CIFAR-100数据集
    # norm_mean = [0.5071, 0.4867, 0.4408]
    # norm_std = [0.2675, 0.2565, 0.2761]
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR100(
    #         'data',                  
    #         train=True,                 
    #         download=True,              
    #         transform=transforms.Compose([         
    #             transforms.ToTensor(),             
    #             transforms.Normalize(
    #                 mean=norm_mean,          
    #                 std=norm_std
    #             )
    #         ])
    #     ),
    #     batch_size=BATCH_SIZE,      
    #     shuffle=True                
    # )
    
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR100(
    #         'data',                    
    #         train=False,               
    #         download=True,              
    #         transform=transforms.Compose([         
    #             transforms.ToTensor(),             
    #             transforms.Normalize(
    #                 mean=norm_mean,          
    #                 std=norm_std
    #             )
    #         ])
    #     ),
    #     batch_size=BATCH_SIZE,       # 每个批次的样本数量
    #     shuffle=False                # 在测试集中一般不需要打乱数据
    # )
    
    # CIFAR-10数据集
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            'data',                    
            train=True,                
            download=True,              
            transform=transforms.Compose([          
                transforms.ToTensor(),             
                transforms.Normalize(mean=norm_mean,         
                std=norm_std)
            ])
        ),
        batch_size=BATCH_SIZE,       
        shuffle=True                
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            'data',                    
            train=False,                
            download=True,              
            transform=transforms.Compose([          
                transforms.ToTensor(),            
                transforms.Normalize(mean=norm_mean,         
                std=norm_std)
            ])
        ),
        batch_size=BATCH_SIZE,      
        shuffle=False              
    )
    
    model = ConvNet()
    
    num_layers = len(list(model.modules())) - 1
    print(f"Number of layers in the model: {num_layers}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
    # model.apply(initialize_weights)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    momentum = 0.9  
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, momentum=momentum)
    optimizer_delta = optim.Adadelta(model.parameters())
    total_loss_adam = []
    total_loss_sgd = []
    total_loss_delta = []
    
    optimizers = [optimizer_adam, optimizer_sgd, optimizer_delta]
    total_losses = [total_loss_adam, total_loss_sgd, total_loss_delta]
    names = ['Adam_10', 'SGD_10', 'Adadelta_10']
    for name, optimizer, total_loss in zip(names, optimizers, total_losses):
        print("Optimizer: ", optimizer)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        for epoch in range(1, EPOCHS + 1):
            train(model, DEVICE, train_loader, optimizer, epoch)
            test_loss = test(model, DEVICE, test_loader)
            scheduler.step()
            total_loss.append(test_loss)
        np.savetxt(f'{name}.txt', total_loss)