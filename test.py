import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils import data
from matplotlib import pyplot as plt
from torchvision import transforms


class CenterLoss(nn.Module):
    def __init__(self, cls_num, feat_num):
        super().__init__()
        self.cls_num = cls_num
        # 中心点定为模型参数(初始值为随机数)
        self.center = nn.Parameter(torch.randn(cls_num, feat_num))

    def forward(self, _x, _y, lamda):
        center_exp = self.center.index_select(dim=0, index=_y.long())
        count = torch.histc(_y.float(), bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_exp = count.index_select(dim=0, index=_y.long())
        # return lamda / 2 * torch.mean(torch.div(torch.sqrt(
        # torch.sum(torch.pow(_x - center_exp, 2), dim=1)), count_exp))
        return lamda / 2 * torch.mean(torch.div(torch.sum(torch.pow((_x - center_exp), 2), dim=1), count_exp))


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, bias=False):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, _x):
        return self.cnn_layer(_x)


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            # ConvLayer(1, 32, 5, 1, 2),
            ConvLayer(3, 32, 5, 1, 2),
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
            # nn.Linear(64, 2)
            nn.Linear(64 * 2 * 2, 2)
        )

        self.output_layer = nn.Sequential(
            # nn.Linear(2, 10)
            nn.Linear(2, 100)
        )

    def forward(self, _x):
        outs = self.hidden_layer(_x)
        # outs = outs.reshape(-1, 64)
        outs = outs.reshape(-1, 64 * 2 * 2)
        feature = self.fc(outs)
        # outs = torch.log_softmax(self.output_layer(feature), dim=1)
        outs = self.output_layer(feature)
        return feature, outs


def visualize(feats, labels, epoch):
    # plt.ion()
    plt.clf()
    color = [
        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555',

        '#DF0029', '#EC870E', '#FCF54C', '#83C75D', '#00B2BF',
        '#426EB4', '#8273B0', '#AF4A92', '#898989', '#555555'
    ]
    for i in range(100):
        plt.plot(feats[labels == i, 0], feats[labels == i, 1], '.', c=color[i])
    plt.legend([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',

        '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
        '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',

        '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
        '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',

        '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
        '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',

        '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
        '90', '91', '92', '93', '94', '95', '96', '97', '98', '99',
    ], loc='upper right')
    plt.title('epoch=%d' % epoch)
    plt.savefig('img4/epoch=%d.jpg' % epoch)
    # plt.pause(0.001)
    # plt.ioff()


if __name__ == '__main__':
    # 测试
    # loss_fn = CenterLoss(5, 2)
    # feat = torch.randn(5, 2, dtype=torch.float32)
    # y_list = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    # loss = loss_fn(feat, y_list, 1)
    # print(loss)
    # tensor(0.2911, grad_fn=<MulBackward0>)

    # net = MainNet()
    # x = torch.randn(1, 1, 28, 28)
    # _out = net(x)
    # print(_out.shape)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_data = torchvision.datasets.MNIST(root=r'E:\pythonProject\datasets\mnist',
    #                                         download=True, train=True, transform=transforms.ToTensor())
    # cifar10_data = torchvision.datasets.CIFAR10(root=r'E:\pythonProject\datasets',
    #                                             download=True, train=True, transform=transforms.ToTensor())
    cifar100_data = torchvision.datasets.CIFAR100(root=r'E:\pythonProject\datasets',
                                                  download=True, train=True, transform=transforms.ToTensor())

    data_loader = data.DataLoader(dataset=cifar100_data, shuffle=True, batch_size=256)
    net = MainNet().to(device=DEVICE)

    loss_fc = nn.CrossEntropyLoss()
    # center_loss_fn = CenterLoss(10, 2).to(device=DEVICE)
    center_loss_fn = CenterLoss(100, 2).to(device=DEVICE)

    net_opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(net_opt, 20, gamma=0.8)
    c_l_opt = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(data_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            feat, out = net(x)

            loss = loss_fc(out, y)
            center_loss = center_loss_fn(feat, y, 1)
            loss += center_loss

            net_opt.zero_grad()
            c_l_opt.zero_grad()
            loss.backward()
            net_opt.step()
            c_l_opt.step()

            feat_loader.append(feat)
            label_loader.append(y)
            if i % 10 == 0:
                print(loss.item(), f'\t{epoch}')
        feats = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        visualize(feats=feats.detach().cpu().numpy(), labels=labels.detach().cpu().numpy(), epoch=epoch)
        scheduler.step()
        epoch += 1
        if epoch == 500:
            break
    pass
