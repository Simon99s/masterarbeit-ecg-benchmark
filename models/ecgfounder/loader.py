import torch
from .net1d import Net1D


def load_ecgfounder(device):

    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150
    )

    checkpoint = torch.load(
        "./models/ecgfounder/checkpoint/12_lead_ECGFounder.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.to(device)
    model.eval()

    return model