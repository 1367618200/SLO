# import torch
# import torch.nn.functional as F
# from mmcls.models.losses.utils import weight_reduce_loss


# predictions = torch.tensor([[0.1,0.2,0.7], [0.2,0.1,0.7],[0.2,0.7,0.1]])# 预测值
# targets = torch.tensor([0, 1,2])# 标签
# weights = torch.tensor([0.3,0.4,0.3 ])# 权重

# loss = F.cross_entropy(predictions, targets, weight=weights, reduction='none')
# loss = weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=len(targets))
# print("函数计算类平衡交叉熵损失:", loss.item())

# predictions_softmax = torch.softmax(predictions, dim=1)
# probs = predictions_softmax[range(len(targets)), targets]# 使用索引操作获取每个样本对应的预测概率
# loss = -torch.mean(weights * torch.log(probs))
# print("直接计算类平衡交叉熵损失:", loss.item())

import cv2

# Load the image
src = cv2.imread('/home/chenqiongpu/SLO/SLO/configs_fundus/SLO/0_study1.jpg')  # Read the image as grayscale
lab_img = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)# 将彩色图像转换为LAB颜色空间
l_channel, a_channel, b_channel = cv2.split(lab_img)# 分割LAB图像的亮度和色度通道
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))# 创建CLAHE对象并应用于亮度通道
l_channel_equalized = clahe.apply(l_channel)
lab_img_equalized = cv2.merge((l_channel_equalized, a_channel, b_channel))# 合并亮度和色度通道
equalized_img = cv2.cvtColor(lab_img_equalized, cv2.COLOR_LAB2BGR)# 将图像转换回BGR颜色空间

# Display the original and processed images
cv2.imshow('Original Image', src)
cv2.imshow('CLAHE Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

