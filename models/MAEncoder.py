import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from models.ViT import *
from models.MADecoder import RightHalfUnet

class ViTMaskAutoEncoder(nn.Module):
    
    def __init__(self,
                img_size=1024,
                patch_size=16,
                embed_dim=768,
                depth=6,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path_rate=0.0,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                use_abs_pos=True,
                use_rel_pos=False,
                rel_pos_zero_init=True,
                window_size=0,
                window_block_indexes=(),
                residual_block_indexes=(),
                use_act_checkpoint=False,
                pretrain_img_size=224,
                pretrain_use_cls_token=True,
                out_feature="last_feat",
                 **kwargs):
        super(ViTMaskAutoEncoder, self).__init__()
        
        self.patch_embed = vit.PatchEmbed(
            kernel_size=patch_size,
            stride=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )
        
        self.vit_block_1 = ViTBlock(
            dim=embed_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias,
            drop=0
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.ViT_id4blocks = nn.ModuleList()
        
        for i in range(depth):
            self.ViT_id4blocks.append(vit.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size)
            ))      
          
        # upsampling: 16x14x14x768 -> 16x28x28x384
        self.up2x = UpSampling(768, 384, 2)
        
        # upsampling: 16x14x14x768 -> 16x56x56x192
        self.up4x = UpSampling(768, 192, 4)
        
        self.up8x = UpSampling(768, 96, 8)
        
        self.up16x = UpSampling(768, 48, 16)
        
        # identity: 16x14x14x768
        self.id = nn.Identity()
        
        # downsampling: 16x14x14x768 -> 16x7x7x1536
        self.down2x = DownSampling(768, 1536, 2)
        
        self.decoder = RightHalfUnet(16)
        
        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            x = x + vit.get_abs_pos(self.pos_embed, True, (x.shape[1], x.shape[2]))
                    
        x = self.vit_block_1(x)
        
        checkpoints = []
        
        for block in self.ViT_id4blocks:
            x = block(x)
            checkpoints.append(torch.permute(x, (0, 3, 1, 2)))
        x_up16x = self.up16x(checkpoints[0])
        x_up8x = self.up8x(checkpoints[1])
        x_up4x = self.up4x(checkpoints[2])
        x_up2x = self.up2x(checkpoints[3])
        x_id = self.id(checkpoints[4])
        x_down2x = self.down2x(checkpoints[5])
        
        x = torch.nested.nested_tensor([x_up16x, x_up8x, x_up4x, x_up2x, x_id, x_down2x])
        
        x = self.decoder(x)
        
        return x