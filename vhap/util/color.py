import torch


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308,
        f * 12.92,
        torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _rgb_to_srgb(f)
    )
    assert (
        out.shape[0] == f.shape[0]
        and out.shape[1] == f.shape[1]
        and out.shape[2] == f.shape[2]
    )
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045,
        f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4),
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _srgb_to_rgb(f)
    )
    assert (
        out.shape[0] == f.shape[0]
        and out.shape[1] == f.shape[1]
        and out.shape[2] == f.shape[2]
    )
    return out


def color_correct_rgb(f: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """
    Apply color correction matrix to linear RGB images.

    Args:
        f: Input image tensor with shape (H, W, 3).
        ccm: Color correction matrix with shape (3, 3).
    """
    assert f.dim() == 3 and f.shape[-1] == 3
    H, W = f.shape[:2]
    out = torch.matmul(f.view(-1, 3), ccm.T).reshape(H, W, 3)
    return out


def color_correct_srgb(f: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """
    Apply color correction matrix to sRGB images.

    Args:
        f: Input image tensor with shape (H, W, 3).
        ccm: Color correction matrix with shape (3, 3).
    """
    assert f.dim() == 3 and f.shape[-1] == 3
    H, W = f.shape[:2]

    out = srgb_to_rgb(f)
    out = torch.matmul(out.view(-1, 3), ccm.T).reshape(H, W, 3).clamp(0.0, 1.0)
    out = rgb_to_srgb(out)
    return out
