
# CenterLoss实现
以MNIST数据集为例，分类模型采用交叉熵损失函数，距离损失使用不带根号版的CenterLoss，收敛速度极快
> 参考文章：[史上最全MNIST系列（三）——Centerloss在MNIST上的Pytorch实现（可视化）](https://www.codenong.com/cs106713478/)


## V1.0
### 中心损失（取消根号版）
```python
return lamda / 2 * torch.mean(torch.div(torch.sum(torch.pow((_x - center_exp), 2), dim=1), count_exp))
```
### 网络结构
```python
self.hidden_layer = nn.Sequential(
    ConvLayer(1, 32, 5, 1, 2),
    ConvLayer(32, 64, 5, 1, 2),
    nn.MaxPool2d(2, 2),
    ConvLayer(64, 128, 5, 1, 2),
    ConvLayer(128, 256, 5, 1, 2),
    nn.MaxPool2d(2, 2),
    ConvLayer(256, 512, 5, 1, 2),
    ConvLayer(512, 512, 5, 1, 2),
    nn.MaxPool2d(2, 2),
    ConvLayer(512, 256, 5, 1, 2),
    ConvLayer(256, 128, 5, 1, 2),
    ConvLayer(128, 64, 5, 1, 2),
    nn.MaxPool2d(2, 2)
)

self.fc = nn.Sequential(
    nn.Linear(64, 2)
)

self.output_layer = nn.Sequential(
    nn.Linear(2, 10)
)
```
### 参数
```python
data_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=256)
# ...
net_opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(net_opt, 20, gamma=0.8)
c_l_opt = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)
```
### 效果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711637911040-c6a71c68-324a-479e-a5d6-6c0b8175ba86.png#averageHue=%23f8f7f5&clientId=u5f5c6e26-9c2c-4&from=paste&height=268&id=ud97559f2&originHeight=241&originWidth=980&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=111546&status=done&style=none&taskId=uce9b480c-b1e5-4744-ae70-c941b7ae09e&title=&width=1088.8889177345943)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711637962750-508af386-99f4-427d-86fb-4aa71ef6bbbf.png#averageHue=%23f9f8f6&clientId=u5f5c6e26-9c2c-4&from=paste&height=360&id=uc036ac3d&originHeight=480&originWidth=640&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=132618&status=done&style=none&taskId=u340f8aaf-371d-4133-93d8-9220b3b8147&title=&width=480)
## V1.1
V1.0基础上，中心损失恢复根号，其余条件不变
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711638108247-c586d533-9503-407b-8aa2-82319a720967.png#averageHue=%232f2e2c&clientId=u5f5c6e26-9c2c-4&from=paste&height=79&id=uf8142aa3&originHeight=71&originWidth=902&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=12261&status=done&style=none&taskId=uf9a04019-1bed-4dcc-9fb1-cff4a779b3a&title=&width=1002.222248772045)
### 中心损失（带根号版）
```python
return lamda / 2 * torch.mean(torch.div(torch.sqrt(torch.sum(torch.pow(_x - center_exp, 2), dim=1)), count_exp))
```
### 效果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711638869142-a99bdae8-9124-46d9-be0a-5721aaf448cb.png#averageHue=%23faf9f8&clientId=u5f5c6e26-9c2c-4&from=paste&height=232&id=u294c4b77&originHeight=209&originWidth=970&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=67423&status=done&style=none&taskId=u18e739f4-2177-4b7d-a9f4-c8589f62193&title=&width=1077.7778063291391)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711638883992-ce578014-0ee6-459f-b403-e7e9647b422c.png#averageHue=%23f8f7f4&clientId=u5f5c6e26-9c2c-4&from=paste&height=360&id=u2a741b32&originHeight=480&originWidth=640&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=173710&status=done&style=none&taskId=ue7c8a71f-cc2c-412e-a226-a81538878ce&title=&width=480)
### 结论
不带根号-->带根号centerloss计算，不带根号的损失计算，让模型分类收敛更快
## V1.2
V1.1基础上，网络结构减少，其余条件不变
### 网络结构
```python
self.hidden_layer = nn.Sequential(
    ConvLayer(1, 32, 5, 1, 2),
    ConvLayer(32, 64, 5, 1, 2),
    nn.MaxPool2d(2, 2),
    ConvLayer(64, 128, 5, 1, 2),
    ConvLayer(128, 256, 5, 1, 2),
    nn.MaxPool2d(2, 2),
    ConvLayer(256, 512, 5, 1, 2),
    ConvLayer(512, 512, 5, 1, 2),
    # nn.MaxPool2d(2, 2),
    # ConvLayer(512, 256, 5, 1, 2),
    # ConvLayer(256, 128, 5, 1, 2),
    # ConvLayer(128, 64, 5, 1, 2),
    nn.MaxPool2d(2, 2)
)

self.fc = nn.Sequential(
    nn.Linear(512 * 3 * 3, 2)
    # nn.Linear(64, 2)
)

self.output_layer = nn.Sequential(
    nn.Linear(2, 10)
)
```
### 效果
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711640079887-0c7497db-e694-454c-8d17-2546a1a6d96e.png#averageHue=%23f9f8f6&clientId=u5f5c6e26-9c2c-4&from=paste&height=252&id=u05ebaa83&originHeight=227&originWidth=972&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=77803&status=done&style=none&taskId=u40ef32a9-bdc7-422b-a61b-42a9c54681e&title=&width=1080.0000286102302)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/38468740/1711640089850-f5ef3c50-a4bf-4f66-b924-0584c35af817.png#averageHue=%23f5f4f0&clientId=u5f5c6e26-9c2c-4&from=paste&height=360&id=ufe3ace64&originHeight=480&originWidth=640&originalType=binary&ratio=0.8999999761581421&rotation=0&showTitle=false&size=199834&status=done&style=none&taskId=u2fb03815-4d19-4430-9d3f-ff624854fa8&title=&width=480)
### 结论
更深的网络层数，收敛速度更快

