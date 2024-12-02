import numpy as np
import cv2
import matplotlib.pyplot as plt


def apply_hough_transform(line_img, min_line_length=10, max_line_gap=20):
    """
    :param line_img
    :param min_line_length
    :param max_line_gap
    rho
    :return
    """
    lines = cv2.HoughLinesP(line_img, rho=1, theta=np.pi/180, threshold=40,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    # print(lines.shape,lines)
    return lines

def output_hough_data(lines):


    angles = []
    rhos = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            dy = y2 - y1
            angle = (np.rad2deg(np.arctan2(dy, dx)) + 180) % 180  
            rho = np.sqrt(dx**2 + dy**2)  
            angles.append(angle)
            rhos.append(rho)

    return angles,rhos

def hough_plot(angle,rho):

    angle = np.array(angle)
    distances = np.array(rho)
    # print(hough_data.shape)

    angles_radians = np.radians(angle)

    # plt.figure(figsize=(8, 8))
    # ax = plt.subplot(111, polar=True)
    # scatter = ax.scatter(angles_radians, distances, c=distances, cmap='viridis', s=50)
    # plt.colorbar(scatter, label="Distance (pixel)")
    # ax.set_theta_zero_location("N")
    # ax.set_theta_direction(-1)
    # plt.title("Angle  Distance in Polar Coordinates", va='bottom')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    scatter = ax.scatter(angles_radians, distances, c=distances, cmap='viridis', edgecolor='k', s=50)
    # ax.set_title("Angle vs Distance in Polar Coordinates", va='bottom')
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Distance (ρ)')
    plt.show()


def generate_histogram_from_hough(lines):

    angle_bins = np.arange(0, 181, 20)  # [0, 20, ..., 160, 180]
    angle_hist = np.zeros(len(angle_bins) - 1)
    list = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算角度（单位：度）
            dx = x2 - x1
            dy = y2 - y1
            angle = (np.rad2deg(np.arctan2(dy, dx)) + 180) % 180  # 将角度限制在 [0, 180)
            rho = np.sqrt(dx**2 + dy**2)  
            list.append((angle,rho))

            for i in range(len(angle_bins) - 1):
                if angle_bins[i] <= angle < angle_bins[i + 1]:
                    lower_weight = (angle_bins[i + 1] - angle) / 20  
                    upper_weight = (angle - angle_bins[i]) / 20  
                    angle_hist[i] += lower_weight * rho
                    if i + 1 < len(angle_hist):
                        angle_hist[i + 1] += upper_weight * rho
                    break

    x_labels = [0,20,40,60,80,100,120,140,160]
    fig, axes = plt.subplots(1)
    bars = axes.bar(x_labels, angle_hist, width=15, color='skyblue',edgecolor='black')
    plt.bar(x_labels, angle_hist, width=15, color='skyblue', edgecolor='black')
    plt.xlim(-10, 170)
    # plt.figure(figsize=(10, 6))
    # plt.bar(angle_bins[:-1], angle_hist, width=18, align='edge', color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Angle(°)")
    plt.ylabel("Cumulative distance (pixel)")
    # plt.xticks(angle_bins)
    for bar in bars:
        y_val = bar.get_height() 
        axes.text(bar.get_x() + bar.get_width() / 2, y_val, f'{y_val:.2f}', ha='center', va='bottom', fontsize=10)
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_lines(img, lines):

    img_with_lines = np.copy(img)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)  
    
    plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    plt.axis('off')  
    plt.savefig('7_sampling_2.png', dpi=300)
    plt.title("Detected Lines")
    plt.show()



image = cv2.imread('7-dot.png')
resized_image = cv2.resize(image, (256, 192))

_, binary_image = cv2.threshold(resized_image[:,:,1], 1, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image, cmap='gray')                                         
plt.axis('off')                                                             
plt.savefig('7_sampling_1.png', dpi=300)
plt.show()
lines = apply_hough_transform(binary_image)                                  

angle,rho = output_hough_data(lines)                                          
hough_plot(angle,rho)                                                        
generate_histogram_from_hough(lines)                                          

color_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

visualize_lines(color_img, lines)













































# 网络模型添加
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class KnowledgeProjectionNet(nn.Module):
#     def __init__(self, external_dim, feature_dim, beta=0.01):
#         super(KnowledgeProjectionNet, self).__init__()
#         self.external_dim = external_dim
#         self.feature_dim = feature_dim
#         self.beta = beta

#         # 定义全连接层和卷积层来学习特征映射
#         self.fc1 = nn.Linear(feature_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(32, external_dim, kernel_size=3, padding=1)

#         # 定义线性映射矩阵 W
#         self.linear_W = nn.Linear(feature_dim, external_dim, bias=False)

#     def forward(self, K, HJ):
#         # 通过全连接层和卷积层处理 HJ
#         x = torch.relu(self.fc1(HJ))
#         x = torch.relu(self.fc2(x))
#         x = x.unsqueeze(1)  # 增加通道维度
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = x.squeeze(1)  # 去掉通道维度

#         # 线性映射矩阵 W
#         HJ_projected = self.linear_W(HJ)

#         return HJ_projected, x

# # 创建一个实例
# external_dim = 64  # 外部知识的维度
# feature_dim = 128  # 模型中间特征的维度
# model = KnowledgeProjectionNet(external_dim, feature_dim)

# # 假设我们有外部知识 K 和中间特征 HJ
# batch_size = 32
# K = torch.randn(batch_size, external_dim)
# HJ = torch.randn(batch_size, feature_dim)

# # 定义损失函数
# def loss_function(K, HJ_projected, W, beta):
#     mse_loss = nn.MSELoss()(HJ_projected, K)
#     l2_reg = beta * torch.norm(W, p=2) ** 2
#     return mse_loss + l2_reg

# # 使用模型进行前向传播
# HJ_projected, W_projection = model(K, HJ)

# # 计算损失
# W = model.linear_W.weight  # 获取权重矩阵 W
# loss = loss_function(K, HJ_projected, W, beta=0.01)

# # 定义优化器
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 进行一个简单的训练步骤
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(f"Loss: {loss.item()}")
