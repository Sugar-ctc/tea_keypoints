import torch
import torch.nn as nn
import torch.optim as optim

class KnowledgeProjectionNet(nn.Module):
    def __init__(self, external_dim, feature_dim, beta=0.01):
        super(KnowledgeProjectionNet, self).__init__()
        self.external_dim = external_dim
        self.feature_dim = feature_dim
        self.beta = beta

        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, external_dim, kernel_size=3, padding=1)

        self.linear_W = nn.Linear(feature_dim, external_dim, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)                                          # (B,H,W,C)
        x = x.reshape(B,-1,C)                                            # (B,H*W,C)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        HJ_projected = x
        x = x.transpose(1,2).view(-1,C,H,W)                        # (B,C,H,W)

        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))

        # x = x.permute(0,2,3,1)                                           # (B,H,W,C)
        # x = x.reshape(B,-1,C)                                            # (B,H*W,C)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))

        # HJ_projected = self.linear_W(x)
        # x = HJ_projected.transpose(1,2).view(-1,self.c1,H,W)             # (B,3,H,W)

        return x, HJ_projected

#     def __init__(self, external_dim, feature_dim, norm_layer=nn.BatchNorm2d):
#         super(KnowledgeProjectionNet, self).__init__()
        
#         self.conv1 = nn.Conv2d_1(
#             external_dim, feature_dim, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv2d_3(
#             external_dim, feature_dim, kernel_size=3, padding=1, bias=False)
#         self.bn = norm_layer(feature_dim, momentum=0.1)
#         self.relu = nn.ReLU(inplace=True)
        

#     def forward(self, x):

#         x = self.conv(x)               # c=3 -> c=128
#         x = self.bn(x)
#         x = self.relu(x)
#         B, C, H, W = x.shape
#         x = x.permute(0,2,3,1)                                          # (B,H,W,C)
#         x = x.reshape(B,-1,C)                                         # (B,13*13,128) 
#         x = torch.relu(self.fc1(x))   
#         x = self.pixel_shuffle(x)
#         return x


#         B, C, H, W = x.shape
#         x = x.permute(0,2,3,1)                                        # (B,H,W,C)
#         x = x.reshape(B,-1,C)                                         # (B,13*13,160) 
#         x = self.fc1(x)                                                 # (B,169,160)->(B,169,256)
#         xin = x.transpose(1,2).view(-1,self.c1,H,W)                     # (B,256,169)->(B,256,13,13)
#         B,C,H,W = xin.shape   



# external_dim = 64  
# feature_dim = 128  
# model = KnowledgeProjectionNet(external_dim, feature_dim)

# batch_size = 32
# K = torch.randn(batch_size, external_dim)
# HJ = torch.randn(batch_size, feature_dim)

# def loss_function(K, HJ_projected, W, beta):
#     mse_loss = nn.MSELoss()(HJ_projected, K)
#     l2_reg = beta * torch.norm(W, p=2) ** 2
#     return mse_loss + l2_reg

# HJ_projected, W_projection = model(K, HJ)


# W = model.linear_W.weight  
# loss = loss_function(K, HJ_projected, W, beta=0.01)


# optimizer = optim.Adam(model.parameters(), lr=0.001)

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(f"Loss: {loss.item()}")
