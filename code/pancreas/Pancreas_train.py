from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result
from torch.distributions import Normal # 用于构建高斯分布进行采样
from Locator import create_Locator # 导入你写的 Locator
import torch.optim as optim

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util import *
from losses import *
from dataloaders import get_ema_model_and_dataloader
import torch.nn.functional as F

"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed_test = 2020
seed_reproducer(seed = seed_test)

data_root, split_name = '../Datasets/pancreas/data', 'pancreas'
result_dir = 'result/pancreas_CARL_V100_GSAM/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 101, 321
pretrain_save_step, st_save_step, pred_step = 10, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

# ======= GSAM3D 核心超参数 =======
use_gsam = True  # 开关：是否在VNet和ResVNet的瓶颈层(Bottleneck)启用GSAM3D全局形状感知模块进行增强
self_train_disable_vnet_gsam = False  # 此开关控制自训练阶段是否关闭 VNet/EMA-VNet 的 GSAM 前向，仅保留 ResVNet 的 GSAM 前向  默认为开启VNet/EMA-VNet以及ResVNet的GSAM前向
# ==============================

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice1 = 0
    max_dice2 = 0
    measures = CutPreMeasures(writer, logger)

    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % 5 == 0:
            net1.eval()
            net2.eval()
            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net_opt(net1, optimizer1, save_path / f'best_ema{label_percent}_pre_vnet.pth', epoch)
                max_dice1 = val_dice1

            if val_dice2 > max_dice2:
                save_net_opt(net2, optimizer2, save_path / f'best_ema{label_percent}_pre_resnet.pth', epoch)
                max_dice2 = val_dice2

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))

        """Training"""
        net1.train()
        net2.train()
        logger.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out1 = net1(img)[0]
            ce_loss1 = F.cross_entropy(out1, lab)
            dice_loss1 = DICE(out1, lab)
            loss1 = (ce_loss1 + dice_loss1) / 2

            out2 = net2(img)[0]
            ce_loss2 = F.cross_entropy(out2, lab)
            dice_loss2 = DICE(out2, lab)
            loss2 = (ce_loss2 + dice_loss2) / 2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            logger.info("cur epoch: %d step: %d" % (epoch, step+1))
            logger.info("vnet")
            measures.update(out1, lab, ce_loss1, dice_loss1, loss1)
            logger.info("resnet")
            measures.update(out2, lab, ce_loss2, dice_loss2, loss2)
            measures.log(epoch, epoch * len(lab_loader_a) + step)


    return max_dice1

def ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, rl_loader, locator, optimizer_locator):

    def get_patches_from_locator(locator_out, raw_images, patch_size=96):
        """
        Args:
            locator_out: [B, 3] 范围在 -1 到 1 之间 (tanh输出)
            raw_images: List of Numpy Arrays (原始尺寸不一)
            patch_size: 裁剪尺寸
        Returns:
            patches_tensor: [B, 1, 96, 96, 96]
            coords_list: 记录实际裁剪坐标，用于调试
        """
        patches = []
        coords_list = []

        # 遍历 Batch 中的每一个样本
        for i, raw_img in enumerate(raw_images):
            # 获取原始图像形状
            shape = raw_img.shape

            # 1. 将 [-1, 1] 映射到 [0, 1]
            norm_coords = (locator_out[i].detach().cpu().numpy() + 1) / 2.0

            # 2. 计算最大允许的起始坐标（防止越界太远）
            # 如果图像本身比 patch 小，max_idx 设为 0
            max_w = max(shape[0] - patch_size, 0)
            max_h = max(shape[1] - patch_size, 0)
            max_d = max(shape[2] - patch_size, 0)

            w1 = int(norm_coords[0] * max_w)
            h1 = int(norm_coords[1] * max_h)
            d1 = int(norm_coords[2] * max_d)

            coords_list.append((w1, h1, d1))

            # 3. 执行裁剪 (Numpy 切片会自动截断超出部分)
            # 例如：如果 w1=100, patch_size=96, 但 shape[0]=150，结果就是 50 长度
            temp_patch = raw_img[w1:w1 + patch_size, h1:h1 + patch_size, d1:d1 + patch_size]

            # 4. 强制 Padding 逻辑 (这是修复报错的关键！！！)
            # 检查切出来的尺寸是否满足 patch_size
            curr_w, curr_h, curr_d = temp_patch.shape

            if curr_w != patch_size or curr_h != patch_size or curr_d != patch_size:
                # 计算需要填充多少
                pad_w = patch_size - curr_w
                pad_h = patch_size - curr_h
                pad_d = patch_size - curr_d

                # 使用 np.pad 进行填充，模式为 constant (填0)
                # 格式: ((前, 后), (前, 后), (前, 后))
                temp_patch = np.pad(temp_patch,
                                    ((0, pad_w), (0, pad_h), (0, pad_d)),
                                    mode='constant',
                                    constant_values=0)

            # 5. 转 Tensor 并增加 Channel 维 [1, D, H, W]
            patch_tensor = torch.from_numpy(temp_patch).float().unsqueeze(0)
            patches.append(patch_tensor)
        return torch.stack(patches).cuda(), coords_list

    def get_XOR_region(mixout1, mixout2):
        s1 = torch.softmax(mixout1, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        s2 = torch.softmax(mixout2, dim = 1)
        l2 = torch.argmax(s2, dim = 1)

        diff_mask = (l1 != l2)
        return diff_mask

    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    load_net_opt(net2, optimizer2, pretrained_path / f'best_ema{label_percent}_pre_resnet.pth')
    load_net_opt(ema_net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    #关闭自训练VNet以及ema_net1的GSAM
    if self_train_disable_vnet_gsam:
        if hasattr(net1, 'use_gsam'):
            net1.use_gsam = False
        if hasattr(ema_net1, 'use_gsam'):
            ema_net1.use_gsam = False
        logger.info('Self-training: disable GSAM forward in VNet and EMA-VNet; keep ResVNet GSAM enabled.')

    max_dice1 = 0
    max_list1 = None
    max_dice2 = 0
    max_dice3 = 0
    total_steps = len(lab_loader_a)  # 用于计算全局步数

    # 将 rl_loader 转为迭代器
    rl_iter = iter(rl_loader)

    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if (epoch % 20 == 0) | ((epoch >= 160) & (epoch % 5 ==0)):

            net1.eval()
            net2.eval()

            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            logger.info('mean average metric is : {}'.format(avg_metric3))

            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]
            val_dice3 = avg_metric3[0]

            # 提取所有指标
            dice1, jaccard1, hd951, asd1 = avg_metric1
            dice2, jaccard2, hd952, asd2 = avg_metric2
            dice3, jaccard3, hd953, asd3= avg_metric3

            # 记录验证指标到TensorBoard
            writer.add_scalar('val/VNet/Dice', dice1, epoch)
            writer.add_scalar('val/VNet/Jaccard', jaccard1, epoch)
            writer.add_scalar('val/VNet/ASD', asd1, epoch)
            writer.add_scalar('val/VNet/95HD', hd951, epoch)

            writer.add_scalar('val/ResNet/Dice', dice2, epoch)
            writer.add_scalar('val/ResNet/Jaccard', jaccard2, epoch)
            writer.add_scalar('val/ResNet/ASD', asd2, epoch)
            writer.add_scalar('val/ResNet/95HD', hd952, epoch)

            writer.add_scalar('val/Mean/Dice', dice3, epoch)
            writer.add_scalar('val/Mean/Jaccard', jaccard3, epoch)
            writer.add_scalar('val/Mean/ASD', asd3, epoch)
            writer.add_scalar('val/Mean/95HD', hd953, epoch)

            if val_dice1 > max_dice1:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice1 = val_dice1
                max_list1 = avg_metric1

            if val_dice2 > max_dice2:
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_resnet.pth'))
                max_dice2 = val_dice2


            if val_dice3 > max_dice3:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self_v.pth'))
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_r.pth'))

                max_dice3 = val_dice3

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))
            logger.info('mean Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice3, max_dice3))

        """Training"""
        net1.train()
        net2.train()
        ema_net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            global_step = (epoch - 1) * total_steps + step

            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out_1 = ema_net1(unimg_a)[0]
                unimg_b_out_1 = ema_net1(unimg_b)[0]

                uimg_a_plab = get_cut_mask(unimg_a_out_1, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out_1, nms=True, connect_mode=connect_mode)


                img_mask, loss_mask = generate_mask(img_a, patch_size)


            """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)

            """BCP"""
            """Supervised Loss"""
            mix_lab_out = net1(net3_input_l)
            mix_output_l = mix_lab_out[0]
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_unlab_out = net1(net3_input_unlab)
            mix_output_2 = mix_unlab_out[0]
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)


            """Supervised Loss"""
            mix_output_l_2 = net2(net3_input_l)[0]
            loss_1_2 = mix_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2_2 = net2(net3_input_unlab)[0]
            loss_2_2 = mix_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask)

            """SDCL"""

            # === Part 2: 新增 FINERS RL 训练代码 ===
            # 1. 获取 RL 数据 (处理迭代器耗尽的情况)
            try:
                rl_batch = next(rl_iter)
            except StopIteration:
                rl_iter = iter(rl_loader)
                rl_batch = next(rl_iter)
            images_global = rl_batch['images_global'].cuda()  # [B, 1, 64, 64, 64]
            images_raw = rl_batch['images_raw']  # List of Numpy
            # 2. Locator 动作预测 (使用策略梯度 REINFORCE)
            mu = locator(images_global)  # [B, 3], range [-1, 1]
            # 构建高斯分布进行采样 (增加探索性)
            # std 可以随着训练衰减，这里先设为固定值 0.1
            std = torch.ones_like(mu) * 0.1
            dist = Normal(mu, std)
            action = dist.sample()  # 采样出的动作，用于裁剪
            # 计算 log_prob，用于更新梯度: loss = -log_prob * reward
            log_prob = dist.log_prob(action).sum(dim=1)
            # 动作截断到 [-1, 1] 用于裁剪 (但梯度回传用 action)
            action_clip = torch.clamp(action, -1, 1)
            # 3. 执行裁剪 (Environment Interaction)
            patches, _ = get_patches_from_locator(action_clip, images_raw, patch_size=96)
            # 4. 放入分割网络 (LPR 阶段)
            # [方案1修改] 让 patches 参与分割网络训练，不再完全冻结梯度

            with torch.no_grad():
                diff_mask1 = get_XOR_region(mix_output_l, mix_output_l_2)
                diff_mask2 = get_XOR_region(mix_output_2, mix_output_2_2)
                # Teacher 网络的输出仍然不需要梯度
                out_teacher = ema_net1(patches)[0]
                r_diff_lab = diff_mask1.float().mean()
                r_diff_unlab = diff_mask2.float().mean()
                r_diff = 0.5 * (r_diff_lab + r_diff_unlab)

            # [方案1核心修改] Student 网络前向传播保留梯度，用于后续分割损失计算
            out_student = net1(patches)[0]
            out_student_net2 = net2(patches)[0]  # net2 也参与训练

            # 5. 计算回顾性奖励 (Retrospective Reward) - 用于 Locator 训练
            # 逻辑：如果 Student 和 Teacher 一致性高，说明裁剪的位置包含了显著的解剖结构
            # 计算一致性损失 (MSE 或 KL)
            # [注意] 这里用 detach() 防止 Locator 的梯度影响分割网络
            consistency_loss_for_reward = softmax_mse_loss(out_student.detach(), out_teacher.detach())
            consistency_loss_for_reward = consistency_loss_for_reward.mean(dim=[1, 2, 3, 4])

            # Reward = exp(-Loss)
            reward = torch.exp(-consistency_loss_for_reward)

            # 6. 更新 Locator
            # Policy Gradient Loss: 最大化 Reward
            locator_loss = -(log_prob * reward).mean()

            optimizer_locator.zero_grad()
            locator_loss.backward()
            optimizer_locator.step()

            # [方案1新增] 计算 patches 的一致性损失，用于训练分割网络
            # 这使得分割网络能够从 Locator 选择的区域中学习
            patch_consistency_loss_net1 = softmax_mse_loss(out_student, out_teacher.detach()).mean()
            patch_consistency_loss_net2 = softmax_mse_loss(out_student_net2, out_teacher.detach()).mean()

            net1_mse_loss_lab = mix_mse_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            # [方案1修改] 将 patch 一致性损失加入总损失，权重 0.1
            loss1 = loss_1 + loss_2 + 0.3 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.1 * (net1_kl_loss_lab + net1_kl_loss_unlab) + 0.1 * patch_consistency_loss_net1

            loss2 = loss_1_2 + loss_2_2 + 0.3 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.1 * (net2_kl_loss_lab + net2_kl_loss_unlab) + 0.1 * patch_consistency_loss_net2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            update_ema_variables(net1, ema_net1, alpha)

            logger.info("loss_1: %.4f, loss_2: %.4f, net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f, patch_cons: %.4f" % 
                (loss_1.item(), loss_2.item(), net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                    net1_kl_loss_lab.item(), net1_kl_loss_unlab.item(), patch_consistency_loss_net1.item()))
            logger.info("r_diff_lab: %.6f, r_diff_unlab: %.6f, r_diff_mean: %.6f" %
                (r_diff_lab.item(), r_diff_unlab.item(), r_diff.item()))

            # 7. 记录日志到 TensorBoard
            writer.add_scalar('train/locator/reward', reward.mean().item(), global_step)
            writer.add_scalar('train/locator/loss', locator_loss.item(), global_step)
            # [方案1新增] 记录 patch 一致性损失
            writer.add_scalar('train/net1/patch_consistency', patch_consistency_loss_net1.item(), global_step)
            writer.add_scalar('train/net2/patch_consistency', patch_consistency_loss_net2.item(), global_step)

            # ======= 新增：记录训练损失到TensorBoard =======
            # global_step = (epoch - 1) * total_steps + step
            # Net1损失
            writer.add_scalar('train/net1/loss_sup', loss_1.item(), global_step)
            writer.add_scalar('train/net1/loss_unsup', loss_2.item(), global_step)
            writer.add_scalar('train/net1/mse_lab', net1_mse_loss_lab.item(), global_step)
            writer.add_scalar('train/net1/mse_unlab', net1_mse_loss_unlab.item(), global_step)
            writer.add_scalar('train/net1/kl_lab', net1_kl_loss_lab.item(), global_step)
            writer.add_scalar('train/net1/kl_unlab', net1_kl_loss_unlab.item(), global_step)
            writer.add_scalar('train/net1/total_loss', loss1.item(), global_step)

            # Net2损失
            writer.add_scalar('train/net2/loss_sup', loss_1_2.item(), global_step)
            writer.add_scalar('train/net2/loss_unsup', loss_2_2.item(), global_step)
            writer.add_scalar('train/net2/mse_lab', net2_mse_loss_lab.item(), global_step)
            writer.add_scalar('train/net2/mse_unlab', net2_mse_loss_unlab.item(), global_step)
            writer.add_scalar('train/net2/kl_lab', net2_kl_loss_lab.item(), global_step)
            writer.add_scalar('train/net2/kl_unlab', net2_kl_loss_unlab.item(), global_step)
            writer.add_scalar('train/net2/total_loss', loss2.item(), global_step)

            # SDCL 分歧比例监控（越低说明两网预测越趋同）
            writer.add_scalar('train/sdcl/r_diff_lab', r_diff_lab.item(), global_step)
            writer.add_scalar('train/sdcl/r_diff_unlab', r_diff_unlab.item(), global_step)
            writer.add_scalar('train/sdcl/r_diff_mean', r_diff.item(), global_step)


        if epoch == self_training_epochs:
            save_net(net1, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
    return max_dice1, max_list1

def test_model(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    load_path = Path(result_dir) / self_train_name
    load_net(net1, load_path / 'best_ema_20_self.pth')
    load_net(net2, load_path / 'best_ema_20_self_resnet.pth')
    print('Successful Loaded')
    avg_metric, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)
    print(avg_metric)
    print(avg_metric2)
    print(avg_metric3)


if __name__ == '__main__':
    try:
        net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, rl_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent, use_gsam=use_gsam)
        # 1. 初始化 Locator
        locator = create_Locator().cuda()
        # Locator 不需要非常大的学习率，通常比分割网络小
        optimizer_locator = optim.Adam(locator.parameters(), lr=1e-4)

        pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader)
        seed_reproducer(seed = seed_test)

        ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader, rl_loader, locator, optimizer_locator)
        test_model(net1, net2, test_loader)

    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


