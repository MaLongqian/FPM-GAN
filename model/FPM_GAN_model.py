import time

import torch

from util.FSDLoss import FSDLoss  # 傅里叶损失
from .base_model import BaseModel
from . import networks
class FPM_GANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # 将默认值更改为与 Pix2Pix 论文（https://phillipi.github.io/pix2pix/）相匹配。
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=200.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # 指定您希望打印的训练损失。训练/测试脚本将调用 <BaseModel.get_current_losses>。
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # 指定您想要保存/显示的图像。训练/测试脚本将调用 <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # 指定您想要保存到磁盘的模型。训练/测试脚本将调用 <BaseModel.save_networks> 和 <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # 定义网络（包括生成器和判别器）
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a  discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.fds = FSDLoss()  # 傅里叶损失
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lrd, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """将输入数据从数据加载器中解包，并执行必要的预处理步骤。

        Parameters:
            input (dict): include the data itself and its metadata information.

        选项 'direction' 可用于交换域 A 和域 B 中的图像。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """“运行前向传播；由两个函数 <optimize_parameters> 和 <test> 调用。”"""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, epoch):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if epoch > 120:
            # alpha = 1.4-0.005*epoch  # L1损失的权重
            # beta = 0.005*epoch-0.4  # 傅里叶损失的权重
            alpha = 1.1-0.003*epoch  # L1损失的权重
            beta = 0.003*epoch-0.1  # 傅里叶损失的权重
        else:
            alpha = 0.8  # L1损失的权重
            beta = 0.2  # 傅里叶损失的权重
        # alpha = 0.803 - 0.003*epoch
        # beta = 0.003*epoch + 0.197

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1  # L1损失

        # self.loss_G_L1 = self.fds(self.fake_B, self.real_B) * self.opt.lambda_L1  # 傅里叶损失

        self.loss_G_L1 = (self.fds(self.fake_B, self.real_B)*beta + self.criterionL1(self.fake_B, self.real_B)*alpha) * self.opt.lambda_L1  # L1+傅里叶损失

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self, epoch=None):
        time_start = time.time()
        epoch = epoch
        self.forward()                   # “计算假图像：G(A)”
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # 在优化 G 时，D 不需要计算梯度。
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(epoch)                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
        # print(time.time()-time_start)



