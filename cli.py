
import argparse

def build_cli():
    p = argparse.ArgumentParser(
        prog='train.py',
        description='Joint Source–Channel Coding experiment launcher'
    )

    # -- 运行模式 --
    p.set_defaults(training=False)                     
    g = p.add_mutually_exclusive_group()
    g.add_argument('-T', '--train', action='store_true',
                   dest='training', help='start training')
    g.add_argument('-E', '--eval',  action='store_false',
                   dest='training', help='run evaluation only')

    # -- 数据集 --
    p.add_argument('--data-train', dest='trainset',
                   default='DIV2K', choices=['CIFAR10', 'DIV2K'],
                   metavar='NAME', help='training corpus')
    p.add_argument('--data-test', dest='testset',
                   default='kodak',
                   choices=['kodak', 'CLIC21', 'div2k', 'afhq'],
                   metavar='NAME', help='benchmark set')

   
    p.add_argument('--metric', dest='distortion_metric',
                   default='MSE', choices=['MSE', 'MS-SSIM'],
                   help='distortion criterion')
    p.add_argument('--arch', dest='model',
                   default='Agent', choices=['Agent', 'Agent_W/O'],
                   help='backbone variant')

    
    p.add_argument('--channel', dest='channel_type',
                   default='awgn', choices=['awgn', 'rayleigh'],
                   help='channel model')
    p.add_argument('-c', '--code-dim', dest='C', type=int,
                   default=96, metavar='N', help='latent (bottleneck) width')
    p.add_argument('--snr', dest='multiple_snr',
                   default='1,4,7,10,13',
                   metavar='LIST', help='comma-separated SNRs in dB')

    args = p.parse_args()

    # 小写/大写统一，防止硬编码比较出错
    args.trainset = args.trainset.upper()
    args.testset  = args.testset.lower()
    return args

  