
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDA_MNet(nn.Module):
    def __init__(self, in_samples=500, n_chans=22, n_cls=4, F1=8, n_convs=3, F2=48, fusion_init_scale=0.9, chans_C=13, chans_P=9):
        super().__init__()
        self.branchC = CLNetStrongMultiViewBranch(n_chans=chans_C, filters=F1, n_convs=n_convs)
        self.branchP = CLNetStrongMultiViewBranch(n_chans=chans_P, filters=F1, n_convs=n_convs)

        self.cross_fusion = CrossStitchUnitPerChannel(channels=F1*n_convs*2, init_scale=fusion_init_scale)
        self.conv_C = nn.Sequential(
                    nn.Conv2d(
                        in_channels=F1*n_convs*2,
                        out_channels=F2,
                        kernel_size=(1, 16),
                        padding='same',
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=F2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 7)),
                    nn.Dropout(0.3)
                )

        self.conv_P = nn.Sequential(
                    nn.Conv2d(
                        in_channels=F1*n_convs*2,
                        out_channels=F2,
                        kernel_size=(1, 16),
                        padding='same',
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=F2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 7)),
                    nn.Dropout(0.3)
                )
        in_shape = int((in_samples // 8 // 7) * F2)

        self.flat = nn.Flatten()

        self.classifier_C = nn.Linear(
                in_features=in_shape,
                out_features=n_cls,
            )

        self.classifier_P = nn.Linear(
                in_features=in_shape,
                out_features=n_cls,
            )


    def forward(self, X_C, X_P):
        feat_C = self.branchC(X_C)
        feat_P = self.branchP(X_P)
        feat_C = feat_C[:,:,-1,:]
        feat_P = feat_P[:,:,-1,:]

        feat_C, feat_P = self.cross_fusion(feat_C, feat_P)
        feat_C = feat_C.unsqueeze(dim=2)
        feat_P = feat_P.unsqueeze(dim=2)
        feat_C = self.conv_C(feat_C)
        feat_P = self.conv_P(feat_P)

        feat_C = self.flat(feat_C)
        feat_P = self.flat(feat_P)

        out_C = self.classifier_C(feat_C)
        out_P = self.classifier_P(feat_P)

        return out_C, feat_C, out_P, feat_P


class CLNetStrongMultiViewBranch(nn.Module):
    def __init__(self, n_chans=22, filters=8, n_convs=3):
        super().__init__()
        self.inception_block = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filters,
                    kernel_size=(1, 64 // (2 ** i)),
                    padding='same',
                ),
                nn.BatchNorm2d(filters),
                GCBlock(in_channels=filters),
                nn.Conv2d(
                    in_channels=filters,
                    groups=filters,
                    out_channels=2 * filters,
                    kernel_size=(n_chans, 1),
                ),
                nn.BatchNorm2d(2 * filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(0.3)
            ) for i in range(n_convs)]
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        feat = [conv(x) for conv in self.inception_block]
        feat = torch.concat(feat, dim=1)
        return feat


class GCBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(GCBlock, self).__init__()
        mid_channels = in_channels//reduction
        self.in_channels = in_channels

        self.reduction = reduction
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(x.size(0), 1, -1)
        context_mask = torch.softmax(context_mask, dim=2)

        context_mask = context_mask.view(x.size(0), 1, x.size(2), x.size(3))
        context = torch.sum(x * context_mask, dim=(2, 3), keepdim=True)
        transform = self.transform(context)

        scale = self.sigmoid(transform)
        out = x + x * scale

        return out


class CrossStitchUnitPerChannel(nn.Module):
    def __init__(self, channels, init_scale=0.9, clamp_range=None):
        super().__init__()
        a11 = torch.full((channels,), init_scale)
        a22 = torch.full((channels,), init_scale)

        off = (1.0 - init_scale) * 0.5
        a12 = torch.full((channels,), off)
        a21 = torch.full((channels,), off)

        alpha_f1 = torch.stack([a11, a12])
        alpha_f2 = torch.stack([a21, a22])

        self.alpha_f1 = nn.Parameter(alpha_f1)
        self.alpha_f2 = nn.Parameter(alpha_f2)
        self.clamp_range = clamp_range


    def forward(self, f1, f2):

        if self.clamp_range is not None:
            with torch.no_grad():
                self.alpha_f1.data.clamp_(min=self.clamp_range[0], max=self.clamp_range[1])
                self.alpha_f2.data.clamp_(min=self.clamp_range[0], max=self.clamp_range[1])

        a11 = self.alpha_f1[0].view(1,-1,1)
        a12 = self.alpha_f1[1].view(1,-1,1)
        a21 = self.alpha_f2[0].view(1,-1,1)
        a22 = self.alpha_f2[1].view(1,-1,1)
        out1 = a11 * f1 + a12 * f2
        out2 = a21 * f1 + a22 * f2
        return out1, out2





