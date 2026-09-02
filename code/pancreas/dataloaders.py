import numpy as np
import torch
import h5py

from scipy.ndimage import zoom # 需要导入 zoom 用于下采样
from torch import import_ir_module, nn as nn, optim as optim
from torch.utils.data import DataLoader
from Vnet import VNet
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from ResVNet import ResVNet

def create_Vnet(ema=False, use_gsam=True):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True, use_gsam=use_gsam)
    # net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
def create_ResNet(ema=False, use_gsam=True):
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True, use_gsam=use_gsam)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

# class RandomCrop(object):
#     """
#     Crop randomly the image in a sample
#     Args:
#     output_size (int): Desired output size
#     """
#
#     def __init__(self, output_size, with_sdf=False):
#         self.output_size = output_size
#         self.with_sdf = with_sdf
#
#     def _get_transform(self, x):
#         if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
#             pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
#             ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
#             pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
#             x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#         else:
#             pw, ph, pd = 0, 0, 0
#
#         (w, h, d) = x.shape
#         w1 = np.random.randint(0, w - self.output_size[0])
#         h1 = np.random.randint(0, h - self.output_size[1])
#         d1 = np.random.randint(0, d - self.output_size[2])
#
#         def do_transform(image):
#             if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
#                 try:
#                     image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#                 except Exception as e:
#                     print(e)
#             image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#             return image
#
#         return do_transform
#
#     def __call__(self, samples):
#         transform = self._get_transform(samples[0])
#         return [transform(s) for s in samples]
class RandomCrop(object):
    """
    修改后的 RandomCrop，支持随机裁剪，也支持指定坐标裁剪
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_random_coords(self, shape):
        # 原始的随机逻辑保留在这里
        w, h, d = shape
        # 增加 padding 逻辑判断，防止负数
        w1 = np.random.randint(0, max(w - self.output_size[0], 1))
        h1 = np.random.randint(0, max(h - self.output_size[1], 1))
        d1 = np.random.randint(0, max(d - self.output_size[2], 1))
        return (w1, h1, d1)

    def crop_at_coords(self, image, coords):
        # 新增：执行特定坐标的裁剪
        w1, h1, d1 = coords
        pw, ph, pd = 0, 0, 0

        # Padding 逻辑 (保持原有逻辑以防越界)
        if image.shape[0] < self.output_size[0] or image.shape[1] < self.output_size[1] or image.shape[2] < \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 1, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # 注意：如果 pad 了，坐标可能需要偏移，这里简化处理，假设 agent 输出的是相对于 pad 后或者原始安全的坐标
        # 为简单起见，这里假设 image 已经足够大或者 Agent 负责处理边界
        patch = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return patch

    def __call__(self, samples, coords=None):
        # 如果传入了 coords，则进行指定裁剪；否则随机裁剪
        image = samples[0]
        if coords is None:
            coords = self._get_random_coords(image.shape)

        return [self.crop_at_coords(s, coords) for s in samples]

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]
    

def get_dataset_path(dataset='pancreas', labelp='10percent'):
    files = ['train_lab.txt', 'train_unlab.txt', 'test.txt']
    return ['../Datasets/pancreas/data_split/{}'.format(f) for f in files]



class Pancreas(Dataset):
    """ Pancreas Dataset """
    # 新增 agent_input_size 参数
    def __init__(self, base_dir, name, split, no_crop=False, labelp=20, reverse=False, mode='standard'):
        self._base_dir = base_dir
        self.split = split
        self.reverse=reverse
        self.mode = mode  # 'standard' (原始SDCL) 或 'rl_train' (引入FINERS)

        #对下述代码进行修改，添加5%标注选项
        # self.labelp = '10percent'
        # if labelp == 20:
        #     self.labelp = '20percent'

        if labelp == 20:
            self.labelp = '20percent'
        elif labelp == 10:
            self.labelp = '10percent'
        elif labelp == 5:
            self.labelp = '5percent'
        else:
            self.labelp = f'{labelp}percent'

        # === 必须添加这两行，否则 __getitem__ 会报错 ===
        self.random_crop = RandomCrop((96, 96, 96))
        self.to_tensor = ToTensor()

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name, self.labelp)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform  # tr_transform
        else:
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    #对下述方法进行修改，增加5%比例的选项
    # def __len__(self):
    #     if self.split == 'train_lab' and self.labelp == '20percent':
    #         return len(self.image_list) * 5
    #     elif self.split == 'train_lab' and self.labelp == '10percent':
    #         return len(self.image_list) * 10
    #     else:
    #         return len(self.image_list)
    def __len__(self):
        if self.split == 'train_lab' and self.labelp == '20percent':
            return len(self.image_list) * 5
        elif self.split == 'train_lab' and self.labelp == '10percent':
            return len(self.image_list) * 10
        elif self.split == 'train_lab' and self.labelp == '5percent':
            # 新增：5% 标注数据需要重复采样 20 次
            return len(self.image_list) * 20
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path+'.h5', 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        # === 核心修改逻辑 ===
        if self.mode == 'rl_train':
            # 模式 A: RL 训练模式
            # 1. 生成给 Agent 看的全局低分辨率图 (假设缩放到 64x64x64)
            # 注意：这里需要根据实际显存调整大小
            scale = np.array([64, 64, 64]) / np.array(image.shape)
            image_global = zoom(image, scale, order=1)  # 线性插值
            # 转换成 Tensor
            image_global = torch.from_numpy(image_global[np.newaxis, ...].astype(np.float32))
            # 2. 返回 原始高分辨率数据 (不裁剪，也不转 Tensor，因为尺寸不一，DataLoader 没法 stack)
            # 我们将在 collate_fn 中处理它们，或者将它们作为 list 返回
            return {
                "image_global": image_global,  # [1, 64, 64, 64]
                "image_raw": image,  # Numpy Array, 尺寸不定
                "label_raw": label,  # Numpy Array, 尺寸不定
                "path": image_path
            }
        else:
            # 模式 B: 原始 SDCL 模式 (Random Crop)
            # 保持原有逻辑，确保预训练代码不受影响
            samples = [image, label]
            # 手动调用 RandomCrop 和 ToTensor，替代原来的 Compose
            # 这样代码逻辑更清晰
            if self.split == 'train_lab' or self.split == 'train_unlab':
                samples = self.random_crop(samples)
            else:
                center_crop = CenterCrop((96, 96, 96))
                samples = center_crop(samples)
            samples = self.to_tensor(samples)
            image_, label_ = samples
            return image_.float(), label_.long()

def rl_collate_fn(batch):
    """
    用于 RL 训练的自定义打包函数
    """
    # 提取全局观测图，它们尺寸是固定的 (64,64,64)，可以 stack
    images_global = torch.stack([item['image_global'] for item in batch])

    # 原始数据尺寸不一，保持为 List
    images_raw = [item['image_raw'] for item in batch]
    labels_raw = [item['label_raw'] for item in batch]
    paths = [item['path'] for item in batch]

    return {
        "images_global": images_global,
        "images_raw": images_raw,
        "labels_raw": labels_raw,
        "paths": paths
    }

def get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=10, use_gsam=True):
    # 根据 label_percent 加载对应的 split 文件
    
    net1 = create_Vnet(use_gsam=use_gsam) #VNet默认开启GSAM，或依据上层控制
    net2 = create_ResNet(use_gsam=use_gsam)

    ema_net1 = create_Vnet(ema=True, use_gsam=use_gsam).cuda()

    optimizer1 = optim.Adam(net1.parameters(), lr=lr)
    optimizer2 = optim.Adam(net2.parameters(), lr=lr)

    trainset_lab_a = Pancreas(data_root, split_name, split='train_lab', labelp=labelp, mode='standard')
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = Pancreas(data_root, split_name, split='train_lab', labelp=labelp, reverse=True, mode='standard')
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    trainset_unlab_a = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp, mode='standard')
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp, reverse=True, mode='standard')
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    # === 新增：RL 专用 Loader ===
    # 注意：batch_size 可以与原来不同，取决于显存。
    # shuffle 必须为 True 以打破相关性
    rl_dataset = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp, mode='rl_train')

    rl_loader = DataLoader(
        rl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=rl_collate_fn,  # 必须使用自定义 collate
        drop_last=True
    )

    return net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, rl_loader