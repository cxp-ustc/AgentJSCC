from models.modules import *
import torch
from models.Agent_encoder import AgentTransformerBlock, AdaptiveModulator

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, out_dim,input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,upsample=None, 
                 agent_num=49, attn_type='A'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        attn_types = [(attn_type if attn_type[0] != 'M' else ('A' if i < int(attn_type[1:]) else 'B')) for i in range(depth)]
        window_sizes = [(window_size if attn_types[i] == 'A' else max(8, (window_size // 8))) for i in range(depth)]
        print( window_sizes)
        self.blocks = nn.ModuleList([
            AgentTransformerBlock(dim=dim,  input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_sizes[i],
                                 shift_size=0 if (i % 2 == 0) else window_sizes[i] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                #  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 agent_num=agent_num,
                                 attn_type=attn_types[i])
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim,out_dim=out_dim,norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops
    
    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            # blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)




class Agent_Decoder(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size, 
                 embed_dims, depths, num_heads,C,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 agent_num=[9, 16, 49, 49], attn_type='BBBB',**kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        # self.num_features = bottleneck_dim
        self.ape = ape
       
        self.patch_norm = patch_norm
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        num_patches = self.H // 4 * self.W // 4

       

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer ]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                            #    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging,
                               agent_num=int(agent_num[i_layer]),
                               attn_type=attn_type[i_layer] + (attn_type[self.num_layers:] if attn_type[i_layer] == 'M' else ''))
            self.layers.append(layer)

        self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[0]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

   

    def forward(self, x, snr, model):
        B, L, C = x.size()
        device = x.get_device()
        x = self.head_list(x)
        if model == 'Agent':
            # token modulation according to input snr value
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

        for i_layer, layer in enumerate(self.layers):
            x = layer(x)

        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
       
        return x

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops
       
    
    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))
            

def create_decoder(**kwargs):
    model = Agent_Decoder(**kwargs)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    return model



def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
