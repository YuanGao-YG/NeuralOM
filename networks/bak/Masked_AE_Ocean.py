import torch
import torch.nn as nn
# from fourier_block import Block
# from utils import get_2d_sincos_pos_embed
from functools import partial
from timm.models.vision_transformer import PatchEmbed
from networks.fourier_block_Masked_AE_Ocean import Block
from networks.t_block_Masked_AE_Ocean import Temporal_Convolution_1d
from networks.basic_modules_Masked_AE_Ocean import get_2d_sincos_pos_embed

class Masked_AFNO(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, params, img_size=(720, 1440), patch_size=(8,8), in_chans=31, out_chans=25,
                 embed_dim=256, depth=24, num_heads=16, time_bolck = False,
                 decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 global_pool=False, drop_path_rate=0., sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0, num_blocks=8, drop_rate = 0.):
        super().__init__()
        # MAE
        self.in_chans = params.in_chans
        self.out_chans = params.out_chans
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = (params.img_size_h, params.img_size_w)
        self.patch_size = (params.patch_size_h, params.patch_size_w)
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]
        self.embed_dim = params.embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_chans, self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, params.depth)]
        
        self.blocks = nn.ModuleList([
            Block(dim=self.embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=self.dpr[i], 
                  norm_layer=norm_layer, num_blocks=params.num_blocks, sparsity_threshold=sparsity_threshold, 
                  hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(params.depth)])
        
        self.norm = norm_layer(self.embed_dim)
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = params.decoder_embed_dim
        
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, params.decoder_depth)]
        
        self.decoder_blocks = nn.ModuleList([
            Block(dim=self.decoder_embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=self.dpr_decoder[i], 
                  norm_layer=norm_layer, num_blocks=params.num_blocks, sparsity_threshold=sparsity_threshold, 
                  hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(params.decoder_depth)])


        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_size[0]*patch_size[1] * self.out_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # new 
        self.global_pool = global_pool
        self.fc_norm = norm_layer(self.embed_dim)
        
        self.L1_error = torch.nn.L1Loss(reduction='mean')
        
        # Time Evolution
        self.time_bolck = time_bolck
        if self.time_bolck:
            self.temporal_evolution = Temporal_Convolution_1d(channel_in = self.embed_dim, channel_hid = self.embed_dim, N_T = 1)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, channel):
        """
        imgs: (N, channel, H, W)
        x: (N, L, patch_size[0]*patch_size[1] * channel)
        """
#         p1 = self.patch_embed.patch_size[0]
#         p2 = self.patch_embed.patch_size[1]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

#         h = imgs.shape[2] // p1
#         w = imgs.shape[3] // p2
        x = imgs.reshape(shape=(imgs.shape[0], channel, self.h, self.patch_size[0], self.w, self.patch_size[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], self.h * self.w, self.patch_size[0]*self.patch_size[1] * channel))
        return x

    def unpatchify(self, x, channel):
        """
        x: (N, L, patch_size[0]*patch_size[1] * 3)
        imgs: (N, 3, H, W)
        """
        assert self.h * self.w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size[0], self.patch_size[1], channel))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channel, self.h * self.patch_size[0], self.w * self.patch_size[1]))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        [1, 256, 1024]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def sequence_masking(self, x, mask_id):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L - 1)

        # sort noise for each sample
        ids = torch.arange(0, L).unsqueeze(0)
        ids_restore = torch.argsort(ids, dim=1)
        ids = ids[:,[i for i in range(x.size(1)) if i != mask_id]]
        ids = ids.expand(N, -1)

        x_masked = torch.gather(x, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, mask_id] = 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    

    def forward_encoder(self, x, mask_ratio):
        # embed patches [1,T,64,64] patchsize
        x = self.patch_embed(x)
        # B = x.shape[0] # [1,T,H,W]

        # add pos embed w/o cls token
        x = x + self.pos_embed
        # [1,T->embeding dimension,H/ps_h * W/ps_w]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # [1,T->embeding dimension,H/ps_h * W/ps_w * mask_ratio]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
#         x = x[:, 1:, :]
        
        # prediction
        x = self.unpatchify(x,self.out_chans)
        return x


    def forward(self, x, mask_ratio=0.0):
        # Patch_Embeding
        x, _, ids_restore = self.forward_encoder(x, mask_ratio)
        
        if self.time_bolck:
            # Time Evolution
            x_t = x.permute([0,2,1])
            x_t = self.temporal_evolution(x_t)
            x_t = x_t.permute([0,2,1])
            x = x + x_t
        
        x = self.forward_decoder(x, ids_restore)  # [N, L, p*p*3]
        return x
