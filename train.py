import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
from models.net import Agent
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
from tqdm import tqdm
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import matplotlib.pyplot as plt
import re
import pandas as pd
import torchvision
parser = argparse.ArgumentParser(description='Agent')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21','div2k','afhq'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='Agent',
                    choices=['Agent', 'Agent_W/O'],
                    help='AgentJSCC model or AgentJSCC without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
args = parser.parse_args()

def validate(net, val_loader, criterion, val_losses):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.cuda()
            outputs = net(inputs)
            # 检查 outputs 是否是元组，如果是则提取第一个元素
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, inputs)  # 使用输入作为目标
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss}')
    val_losses.append(val_loss)
    return val_loss

save_model_path = "./test"
test_result_path = "./test_result"
class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/Dataset/CIFAR10/"
        test_data_dir = "/media/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["/home/chuxiaopeng/media/Dataset/HR_Image_dataset/"]
        if args.testset == 'kodak':
            test_data_dir = ["/home/chuxiaopeng/media/Dataset/kodak_test/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/home/chuxiaopeng/media/Dataset/CLIC21/"]
        elif args.testset == 'afhq':
            test_data_dir = ["/home/chuxiaopeng/media/Dataset/afhq/"]
        else:
            test_data_dir = ["/home/chuxiaopeng/media/Dataset/DIV2K_valid_HR/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,agent_num=[9, 16, 49, 49], attn_type='AAAA',
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,agent_num=[49, 49,16,9], attn_type='AAAA',
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def train_one_epoch(args, net, train_loader, optimizer, criterion, epoch, cur_lr, config):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in tqdm(enumerate(train_loader)):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - loss_G
                msssims.update(msssim)
                # msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                # msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                # logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

def test(epoch=0):
    if epoch==0:
        model_path = f"{save_model_path}/Agent_AWGN_DIV2K_fixed_random_snr_psnr_c{args.C}.model"
        # model_path = f"/home/chuxiaopeng/Agent/AGENT1/Agent_AWGN_DIV2K_fixed_random_snr_psnr_c{args.C}.model"
        load_weights(model_path)

    # count=0
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))

    data = {
        "SNR": [],
        "CBR": [],
        "PSNR": [],
        "MS-SSIM": []
    }
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in tqdm(enumerate(test_loader)):
                    # count+=1
                    start_time = time.time()
                    input = input.cuda()
                    # filename = f"batch{count}.png"
                    # filepath1 = os.path.join("/home/chuxiaopeng/Agent/original", filename)
                    # torchvision.utils.save_image(input,filepath1)
                    print(input.size())
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    # filename = f"batch{count}.png"
                    # filepath = os.path.join("/home/chuxiaopeng/Agent/reverse_result_rayleigh", filename)
                    # torchvision.utils.save_image(recon_image,filepath)
                    # print(SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    # logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        # results_msssim[i]=-10 * np.log10(1 - results_msssim[i])
        for t in metrics:
            t.clear()
        
        # 保存结果到字典
        data["SNR"].append(results_snr[i])
        data["CBR"].append(results_cbr[i])
        data["PSNR"].append(results_psnr[i])
        data["MS-SSIM"].append(results_msssim[i])
    

    # 将结果保存到 Excel 文件
    save_path = f"{test_result_path}/test_results_epoch_{epoch}.xlsx"
    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = Agent(args, config)
    # model_path = f"{save_model_path}/Agent_AWGN_DIV2K_fixed_random_snr_psnr_c{args.C}.model"
    # # # model_path = f"/home/chuxiaopeng/Agent/AGENT1/Agent_AWGN_DIV2K_fixed_random_snr_psnr_c{args.C}.model"
    # load_weights(model_path)
    criterion = torch.nn.MSELoss()  # 使用合适的损失函数
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)

    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    # steps_epoch=1500
    min_delta = 0.00001
    early_stop_patience = 3000
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            print("epoch: ", epoch)
            train_one_epoch(args, net, train_loader, optimizer, criterion, epoch, cur_lr, config)
            val_loss = validate(net, test_loader, criterion, val_losses)
            # if val_loss < best_val_loss :
            #     best_val_loss = val_loss
            #     modelname = f'Agent_{args.channel_type.upper()}_{args.trainset}_fixed_random_snr_psnr_c{args.C}'
            #     modelname = re.sub(r'[<>:"/\\|?*]', '_', modelname)  # 额外添加
            #     save_path = os.path.join(save_model_path, '{}.model'.format(modelname))
            #     save_model(net, save_path)
            #     print('Saving model at epoch:', epoch + 1)
            #     test(epoch)
            
            if epoch%10==0 :
                modelname = f'Agent_{args.channel_type.upper()}_{args.trainset}_fixed_random_snr_psnr_c{args.C}'
                modelname = re.sub(r'[<>:"/\\|?*]', '_', modelname)  # 额外添加
                save_path = os.path.join(save_model_path, '{}.model'.format(modelname))
                save_model(net, save_path)
                print('Saving model at epoch:', epoch + 1)
                test(epoch)
                
    else:
        test()



