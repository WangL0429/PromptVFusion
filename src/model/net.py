# MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan
# ECCV 2026
#
# The RAFT inference wrapper (RAFT / RAFT_component) is adapted from SEA-RAFT
# (Princeton-VL, https://github.com/princeton-vl/SEA-RAFT) and the engineering
# scaffolding is derived from UniVF (Zixiang Zhao et al., NeurIPS 2025 Spotlight).

import torch
import torch.nn as nn
import torch.nn.functional as F
from .raft import RAFT
from .RAFT_component.raft_utils import load_ckpt
from .fusion import *
from .utils import load_args_from_json, flow_warp


class LightweightAAM(nn.Module):
    def __init__(self, dim, base_channels=32, current_flow_mode='legacy_mean'):
        super().__init__()
        self.dim = dim
        if current_flow_mode not in ('zero', 'legacy_mean'):
            raise ValueError('current_flow_mode must be zero or legacy_mean')
        # A plain attribute keeps the original checkpoint parameter structure.
        self.current_flow_mode = current_flow_mode

        # Lightweight refinement encoder (DWConv + PWConv)
        mid_channels = base_channels
        self.compress = nn.Conv2d(dim * 4 + 4, mid_channels, 1)  # 1x1 channel reduction

        self.refine_ds_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 2, 3, 1, 1)  # final residual flow
        )

        # Lightweight temporal aggregation (softmax-weighted sum, replacing Conv3d)
        self.temp_weight = nn.Parameter(torch.ones(1, 1, 3, 1, 1))

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )

        # Static branch fallback
        self.static_refine = nn.Conv2d(dim, dim, 1) if dim > 16 else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, f_curr, f_prev, f_next, flow_prev, flow_next, motion_mask, f_anchor):
        # A: Initial warp
        f_prev_a = flow_warp(f_prev, flow_prev)
        f_next_a = flow_warp(f_next, flow_next)
        if self.current_flow_mode == 'zero':
            # Current features already occupy the output grid. With two past
            # neighbors, their mean displacement is not a current-frame flow.
            # Keep learned residual refinement below; only its base is zero.
            flow_init = torch.zeros_like(flow_prev)
        elif self.current_flow_mode == 'legacy_mean':
            flow_init = (flow_prev + flow_next) * 0.5
        else:
            raise ValueError('current_flow_mode must be zero or legacy_mean')
        f_curr_a = flow_warp(f_curr, flow_init)

        # B: Flow refinement
        res_in = torch.cat([f_anchor, f_prev_a, f_curr_a, f_next_a, flow_prev, flow_next], dim=1)
        res_feat = self.compress(res_in)
        flow_res = self.refine_ds_conv(res_feat)

        f_curr_final = flow_warp(f_curr, flow_init + flow_res)

        # C: Temporal aggregation (softmax-weighted, memory-efficient)
        st_feat = torch.stack([f_prev_a, f_curr_final, f_next_a], dim=2)
        feat_aggregated = (st_feat * torch.softmax(self.temp_weight, dim=2)).sum(dim=2)

        # D: Spatial refinement + motion-mask gating
        feat_refined = self.spatial_refine(feat_aggregated)

        f_static = self.static_refine(f_curr)
        out = feat_refined * motion_mask + f_static * (1 - motion_mask)

        return self.act(out)


class VideoFusion(nn.Module):
    def __init__(self, model_config):
        super(VideoFusion, self).__init__()

        self.causal = bool(model_config['model'].get('causal', False))
        requested_mode = model_config['model'].get(
            'causal_current_flow', 'zero' if self.causal else 'legacy_mean')
        if requested_mode not in ('zero', 'legacy_mean'):
            raise ValueError('causal_current_flow mode must be zero or legacy_mean')
        # Preserve centered behavior; the new base applies to causal models.
        self.causal_current_flow = requested_mode if self.causal else 'legacy_mean'

        # 1. Optical flow network (SEA-RAFT, frozen during training)
        self.raft_args = load_args_from_json("config/module/spring-S.json")
        self.flow_net = RAFT(self.raft_args).eval()
        load_ckpt(self.flow_net, self.raft_args.path)

        # Freeze RAFT parameters
        for param in self.flow_net.parameters():
            param.requires_grad = False

        dim = model_config['model']['dim']
        # Feature-only callers share this exact pretrained encoding space.
        # This attribute introduces no checkpoint parameter or buffer.
        self.feature_dim = dim
        # Shallow feature extractor (3 -> dim) for both IR and VI modalities
        self.feat_extractor_ir = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 3, 1, 1)
        )
        self.feat_extractor_vi = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 3, 1, 1)
        )
        self.aam_ir = LightweightAAM(dim, current_flow_mode=self.causal_current_flow)
        self.aam_vi = LightweightAAM(dim, current_flow_mode=self.causal_current_flow)
        self.fusion_net = Fusion_Net(dim=model_config['model']['dim'])

    def encode_modality(self, image, modality):
        """Encode a real observed image (or loss-only teacher target).

        Images are BCHW in [0,1]. Infrared may be luminance or replicated RGB.
        This method does not choose availability or load any hidden images;
        callers keep observed encoding and teacher-target encoding separate.
        """
        if modality not in ('visible', 'infrared'):
            raise ValueError('Unknown modality for feature encoding')
        channels = (3,) if modality == 'visible' else (1, 3)
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or image.shape[1] not in channels or min(image.shape[-2:]) < 1
                or not image.is_floating_point() or not torch.isfinite(image).all()
                or (image < 0).any() or (image > 1).any()):
            raise ValueError('Observed image must be finite BCHW in [0,1] with valid channels')
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        encoder = self.feat_extractor_vi if modality == 'visible' else self.feat_extractor_ir
        return encoder(image)

    def forward_features(self, ir_features, vi_features, flow_prev, flow_old, motion_mask):
        """Fuse resolved full-resolution features without RGB completion/RAFT.

        Both feature windows are [B,T,D,H,W], T>=3, D=feature_dim. Offsets are
        target(current)-to-source(past) in feature-grid pixels and are supplied
        by the caller's feature correspondence module. Only t-2,t-1,t are used.
        The only image this path produces is the final fused [B,1,3,H,W].
        """
        if (not isinstance(ir_features, torch.Tensor) or not isinstance(vi_features, torch.Tensor)
                or ir_features.ndim != 5 or ir_features.shape != vi_features.shape
                or ir_features.shape[1] < 3 or ir_features.shape[2] != self.feature_dim):
            raise ValueError('Expected matched [B,T,D,H,W] feature windows with T>=3')
        batch, _, _, height, width = ir_features.shape
        if min(height, width) < 2:
            raise ValueError('Feature dimensions must support spatial warping')
        for value in (ir_features, vi_features, flow_prev, flow_old, motion_mask):
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.device != ir_features.device or value.dtype != ir_features.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('Features, offsets and mask must share floating dtype/device and be finite')
        if (tuple(flow_prev.shape) != (batch, 2, height, width)
                or tuple(flow_old.shape) != (batch, 2, height, width)
                or tuple(motion_mask.shape) != (batch, 1, height, width)
                or (motion_mask < 0).any() or (motion_mask > 1).any()):
            raise ValueError('Offsets must be B2HW and the soft mask B1HW in [0,1]')
        return self._fuse_causal_feature_triplets(
            tuple(ir_features[:, j] for j in (-3, -2, -1)),
            tuple(vi_features[:, j] for j in (-3, -2, -1)),
            flow_prev, flow_old, motion_mask)

    def get_flow(self, img1, img2, max_res=256):
        """
        Efficient optical-flow inference (adaptive resolution).

        Downscales inputs to ``max_res`` on the longer side, runs SEA-RAFT, then
        upsamples the flow back to the original resolution (and rescales
        magnitudes by ``1/scale``).
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            b, c, h, w = img1.shape

            scale_factor = min(max_res / max(h, w), 1.0)
            if scale_factor < 1.0:
                new_size = (int(h * scale_factor), int(w * scale_factor))
                imgs = torch.cat([img1, img2], dim=0)
                imgs_resized = F.interpolate(imgs, size=new_size, mode='bilinear', align_corners=False)
                img1_resized, img2_resized = imgs_resized.chunk(2, dim=0)
            else:
                img1_resized, img2_resized = img1, img2

            # Run SEA-RAFT (expects [0, 255] range)
            flow_result = self.flow_net(img1_resized * 255.0, img2_resized * 255.0, test_mode=True)

            # Extract the predicted flow tensor
            if isinstance(flow_result, dict):
                flow = flow_result.get('final', list(flow_result.values())[-1])
            elif isinstance(flow_result, (list, tuple)):
                flow = flow_result[-1]
            else:
                flow = flow_result

            # If we downscaled, upsample back and rescale magnitudes
            if scale_factor < 1.0:
                flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
                flow = flow * (1.0 / scale_factor)

            return flow

    def generate_motion_mask_from_flow(self, flow_prev, flow_next):
        """
        Build a binary motion mask from forward / backward optical flow magnitudes.

        The mask is True where either of the two flows has magnitude > 0.5 px
        (followed by 3x3 max-pool dilation) and serves as a physical prior for
        the MDIM module to allocate sparse attention to moving regions.
        """
        with torch.no_grad():
            mag_prev = torch.sqrt(flow_prev[:, 0] ** 2 + flow_prev[:, 1] ** 2 + 1e-6)
            mag_next = torch.sqrt(flow_next[:, 0] ** 2 + flow_next[:, 1] ** 2 + 1e-6)
            mag = torch.max(mag_prev, mag_next)

            mask = (mag > 0.5).float().unsqueeze(1)
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)  # 3x3 dilation

        return mask

    def _forward_causal(self, sources_1, sources_2):
        """Fuse the current (last) frame using only current and past inputs.

        Both inputs are [B,T,3,H,W] in [0,1], T>=3. The first two
        positions of a five-frame window are reserved for the upstream
        four-frame Motion Graph imputer; MAVFusion itself uses t-2,t-1,t.
        Return [B,1,3,H,W] so the time dimension remains explicit.
        """
        if sources_1.ndim != 5 or sources_1.shape != sources_2.shape or sources_1.shape[1] < 3:
            raise ValueError('Causal MAVFusion requires matched [B,T,C,H,W] inputs with T>=3')
        ir_old, ir_prev, ir_curr = (sources_1[:, j] for j in (-3, -2, -1))
        vi_old, vi_prev, vi_curr = (sources_2[:, j] for j in (-3, -2, -1))

        f_ir_old = self.feat_extractor_ir(ir_old)
        f_ir_prev = self.feat_extractor_ir(ir_prev)
        f_ir_curr = self.feat_extractor_ir(ir_curr)
        f_vi_old = self.feat_extractor_vi(vi_old)
        f_vi_prev = self.feat_extractor_vi(vi_prev)
        f_vi_curr = self.feat_extractor_vi(vi_curr)

        # flow_warp samples past features on the current grid: RAFT must
        # estimate current -> past. Do not reuse the original centered
        # branch's cached future flow across frame pairs. When VI is
        # imputed, these flows are necessarily estimated on predicted VI.
        flow_prev = self.get_flow(vi_curr, vi_prev)
        flow_old = self.get_flow(vi_curr, vi_old)
        motion_mask = self.generate_motion_mask_from_flow(flow_prev, flow_old)

        return self._fuse_causal_feature_triplets(
            (f_ir_old, f_ir_prev, f_ir_curr), (f_vi_old, f_vi_prev, f_vi_curr),
            flow_prev, flow_old, motion_mask)

    def _fuse_causal_feature_triplets(self, infrared, visible, flow_prev, flow_old, motion_mask):
        """Shared feature fusion; the image API supplies its encoded features."""
        f_ir_old, f_ir_prev, f_ir_curr = infrared
        f_vi_old, f_vi_prev, f_vi_curr = visible
        f_ir_aligned = self.aam_ir(
            f_curr=f_ir_curr, f_prev=f_ir_prev, f_next=f_ir_old,
            flow_prev=flow_prev, flow_next=flow_old,
            motion_mask=motion_mask, f_anchor=f_vi_curr,
        )
        f_vi_aligned = self.aam_vi(
            f_curr=f_vi_curr, f_prev=f_vi_prev, f_next=f_vi_old,
            flow_prev=flow_prev, flow_next=flow_old,
            motion_mask=motion_mask, f_anchor=f_ir_curr,
        )
        fused_frame = self.fusion_net(
            f_ir_aligned + f_ir_curr, f_vi_aligned + f_vi_curr, motion_mask,
        )
        return fused_frame.unsqueeze(1)

    def forward(self, sources_1, sources_2):
        if getattr(self, 'causal', False):
            return self._forward_causal(sources_1, sources_2)
        b, t, c, h, w = sources_1.shape
        fused_results_list = []
        cached_flow_next = None

        for i in range(1, t - 1):
            # Sliding window: previous, current, next frames
            ir_prev, ir_curr, ir_next = sources_1[:, i - 1], sources_1[:, i], sources_1[:, i + 1]
            vi_prev, vi_curr, vi_next = sources_2[:, i - 1], sources_2[:, i], sources_2[:, i + 1]

            # Stage 1: shallow feature extraction
            f_ir_curr = self.feat_extractor_ir(ir_curr)
            f_ir_prev = self.feat_extractor_ir(ir_prev)
            f_ir_next = self.feat_extractor_ir(ir_next)

            f_vi_curr = self.feat_extractor_vi(vi_curr)
            f_vi_prev = self.feat_extractor_vi(vi_prev)
            f_vi_next = self.feat_extractor_vi(vi_next)

            # Stage 2: optical flow + motion mask (cached across consecutive frames)
            if cached_flow_next is not None:
                flow_prev = cached_flow_next
            else:
                flow_prev = self.get_flow(vi_prev, vi_curr)
            flow_next = self.get_flow(vi_next, vi_curr)
            cached_flow_next = flow_next

            motion_mask = self.generate_motion_mask_from_flow(flow_prev, flow_next)

            # Stage 3: motion-aware alignment (cross-modal anchor)
            f_ir_final = self.aam_ir(
                f_curr=f_ir_curr, f_prev=f_ir_prev, f_next=f_ir_next,
                flow_prev=flow_prev, flow_next=flow_next,
                motion_mask=motion_mask, f_anchor=f_vi_curr
            )

            f_vi_final = self.aam_vi(
                f_curr=f_vi_curr, f_prev=f_vi_prev, f_next=f_vi_next,
                flow_prev=flow_prev, flow_next=flow_next,
                motion_mask=motion_mask, f_anchor=f_ir_curr
            )

            f_ir_final = f_ir_final + f_ir_curr
            f_vi_final = f_vi_final + f_vi_curr

            # Stage 4: dual-interaction fusion
            fused_frame = self.fusion_net(f_ir_final, f_vi_final, motion_mask)

            fused_results_list.append(fused_frame)

        return torch.stack(fused_results_list, dim=1)
