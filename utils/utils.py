import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def interp(x, xp, fp):
    """
    Linear interpolation of values fp from known points xp to new points x.
    """
    indices = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1 = xp[indices - 1], xp[indices]
    y0, y1 = fp[indices - 1], fp[indices]
    
    return (y0 + (y1 - y0) * (x - x0) / (x1 - x0)).to(fp.dtype)

def multi_interp(x, xp, fp):
    """
    Multi-dimensional linear interpolation of fp from xp to x along all axes.
    """
    if torch.is_tensor(fp):
        out = [interp(x, xp, fp[:, i]) for i in range(fp.shape[-1])]
        return torch.stack(out, dim=-1).to(fp.dtype)
    else:
        out = [np.interp(x, xp, fp[:, i]) for i in range(fp.shape[-1])]
        return np.stack(out, axis=-1).astype(fp.dtype)

def interpolate_params(params, t):
    """
    Interpolate parameters over time t, linearly between frames.
    """
    num_frames = params.shape[-1]
    frame_number = t * (num_frames - 1)
    integer_frame = torch.floor(frame_number).long()
    fractional_frame = frame_number.to(params.dtype) - integer_frame.to(params.dtype)

    # Ensure indices are within valid range
    next_frame = torch.clamp(integer_frame + 1, 0, num_frames - 1)
    integer_frame = torch.clamp(integer_frame, 0, num_frames - 1)

    param_now = params[:, :, integer_frame]
    param_next = params[:, :, next_frame]

    # Linear interpolation between current and next frame parameters
    interpolated_params = param_now + fractional_frame * (param_next - param_now)

    return interpolated_params.squeeze(0).squeeze(-1).permute(1, 0)

class MaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, encoding, mask_coef, initial=0.4):
        """
        Forward pass for masking, scales mask_coef to blend encoding.
        """
        mask_coef = initial + (1 - initial) * mask_coef
        mask = torch.zeros_like(encoding[0:1])
        mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
        mask[..., :mask_ceil] = 1.0
        ctx.save_for_backward(mask)
        return encoding * mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to retain masked gradients.
        """
        mask, = ctx.saved_tensors
        return grad_output * mask, None, None

def mask(encoding, mask_coef, initial=0.4):
    """
    Apply mask to encoding with scaling factor mask_coef.
    """
    return MaskFunction.apply(encoding, mask_coef, initial)

def make_grid(height, width, u_lims, v_lims):
    """
    Create (u,v) meshgrid of size (height, width) with given limits.
    """
    u = torch.linspace(u_lims[0], u_lims[1], width)
    v = torch.linspace(v_lims[1], v_lims[0], height)  # Flip for array convention
    u_grid, v_grid = torch.meshgrid([u, v], indexing="xy")
    return torch.stack((u_grid.flatten(), v_grid.flatten())).permute(1, 0)

def unwrap_quaternions(q):
    """
    Remove 2pi wraps from quaternion rotations.
    """
    n = q.shape[0]
    unwrapped_q = q.clone()
    for i in range(1, n):
        cos_theta = torch.dot(unwrapped_q[i-1], unwrapped_q[i])
        if cos_theta < 0:
            unwrapped_q[i] = -unwrapped_q[i]
    return unwrapped_q

@torch.jit.script
def quaternion_conjugate(q):
    """
    Return the conjugate of a quaternion.
    """
    q_conj = q.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]  # Invert vector part
    return q_conj

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    """
    w1, v1 = q1[..., 0], q1[..., 1:]
    w2, v2 = q2[..., 0], q2[..., 1:]
    w = w1 * w2 - torch.sum(v1 * v2, dim=-1)
    v = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + torch.cross(v1, v2, dim=-1)
    return torch.cat((w.unsqueeze(-1), v), dim=-1)

@torch.jit.script
def convert_quaternions_to_rot(quaternions):
    """
    Convert quaternions (WXYZ) to 3x3 rotation matrices.
    """
    w, x, y, z = quaternions.unbind(-1)
    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x**2 + y**2)
    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return R.reshape(-1, 3, 3)

@torch.no_grad()
def raw_to_rgb(bundle):
    """
    Convert RAW mosaic to three-channel RGB volume by filling empty pixels.
    """
    raw_frames = torch.tensor(np.array([bundle[f'raw_{i}']['raw'] for i in range(bundle['num_raw_frames'])]).astype(np.int32), dtype=torch.float32)[None]
    raw_frames = raw_frames.permute(1, 0, 2, 3)
    color_correction_gains = bundle['raw_0']['color_correction_gains']
    color_filter_arrangement = bundle['characteristics']['color_filter_arrangement']
    blacklevel = torch.tensor(np.array([bundle[f'raw_{i}']['blacklevel'] for i in range(bundle['num_raw_frames'])]))[:, :, None, None]
    whitelevel = torch.tensor(np.array([bundle[f'raw_{i}']['whitelevel'] for i in range(bundle['num_raw_frames'])]))[:, None, None, None]
    shade_maps = torch.tensor(np.array([bundle[f'raw_{i}']['shade_map'] for i in range(bundle['num_raw_frames'])])).permute(0, 3, 1, 2)
    shade_maps = F.interpolate(shade_maps, size=(raw_frames.shape[-2]//2, raw_frames.shape[-1]//2), mode='bilinear', align_corners=False)

    top_left = raw_frames[:, :, 0::2, 0::2]
    top_right = raw_frames[:, :, 0::2, 1::2]
    bottom_left = raw_frames[:, :, 1::2, 0::2]
    bottom_right = raw_frames[:, :, 1::2, 1::2]

    if color_filter_arrangement == 0:  # RGGB
        R, G1, G2, B = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 1:  # GRBG
        G1, R, B, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 2:  # GBRG
        G1, B, R, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 3:  # BGGR
        B, G1, G2, R = top_left, top_right, bottom_left, bottom_right

    # Apply color correction gains, flip to portrait
    R = ((R - blacklevel[:, 0:1]) / (whitelevel - blacklevel[:, 0:1]) * color_correction_gains[0])
    R *= shade_maps[:, 0:1]
    G1 = ((G1 - blacklevel[:, 1:2]) / (whitelevel - blacklevel[:, 1:2]) * color_correction_gains[1])
    G1 *= shade_maps[:, 1:2]
    G2 = ((G2 - blacklevel[:, 2:3]) / (whitelevel - blacklevel[:, 2:3]) * color_correction_gains[2])
    G2 *= shade_maps[:, 2:3]
    B = ((B - blacklevel[:, 3:4]) / (whitelevel - blacklevel[:, 3:4]) * color_correction_gains[3])
    B *= shade_maps[:, 3:4]

    rgb_volume = torch.zeros(raw_frames.shape[0], 3, raw_frames.shape[-2], raw_frames.shape[-1], dtype=torch.float32)

    # Fill gaps in blue channel
    rgb_volume[:, 2, 0::2, 0::2] = B.squeeze(1)
    rgb_volume[:, 2, 0::2, 1::2] = (B + torch.roll(B, -1, dims=3)).squeeze(1) / 2
    rgb_volume[:, 2, 1::2, 0::2] = (B + torch.roll(B, -1, dims=2)).squeeze(1) / 2
    rgb_volume[:, 2, 1::2, 1::2] = (B + torch.roll(B, -1, dims=2) + torch.roll(B, -1, dims=3) + torch.roll(B, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # Fill gaps in green channel
    rgb_volume[:, 1, 0::2, 0::2] = G1.squeeze(1)
    rgb_volume[:, 1, 0::2, 1::2] = (G1 + torch.roll(G1, -1, dims=3) + G2 + torch.roll(G2, 1, dims=2)).squeeze(1) / 4
    rgb_volume[:, 1, 1::2, 0::2] = (G1 + torch.roll(G1, -1, dims=2) + G2 + torch.roll(G2, 1, dims=3)).squeeze(1) / 4
    rgb_volume[:, 1, 1::2, 1::2] = G2.squeeze(1)

    # Fill gaps in red channel
    rgb_volume[:, 0, 0::2, 0::2] = R.squeeze(1)
    rgb_volume[:, 0, 0::2, 1::2] = (R + torch.roll(R, -1, dims=3)).squeeze(1) / 2
    rgb_volume[:, 0, 1::2, 0::2] = (R + torch.roll(R, -1, dims=2)).squeeze(1) / 2
    rgb_volume[:, 0, 1::2, 1::2] = (R + torch.roll(R, -1, dims=2) + torch.roll(R, -1, dims=3) + torch.roll(R, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    rgb_volume = torch.flip(rgb_volume.transpose(-1, -2), [-1])  # Rotate 90 degrees to portrait
    return rgb_volume

def de_item(bundle):
    """
    Call .item() on all dictionary items, removing extra dimensions.
    """
    bundle['motion'] = bundle['motion'].item()
    bundle['characteristics'] = bundle['characteristics'].item()
    
    for i in range(bundle['num_raw_frames']):
        bundle[f'raw_{i}'] = bundle[f'raw_{i}'].item()

def debatch(batch):
    """
    Collapse batch and channel dimensions together.
    """
    debatched = []
    for x in batch:
        if len(x.shape) <= 1:
            raise Exception("This tensor is too small to debatch.")
        elif len(x.shape) == 2:
            debatched.append(x.reshape(x.shape[0] * x.shape[1]))
        else:
            debatched.append(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]))
    return debatched

def apply_ccm(image, ccm):
    """
    Apply Color Correction Matrix (CCM) to the image.
    """
    if image.dim() == 3:
        corrected_image = torch.einsum('ij,jkl->ikl', ccm, image)
    else:
        corrected_image = ccm @ image
    return corrected_image.clamp(0, 1)

def apply_tonemap(image, tonemap):
    """
    Apply tonemapping curve to the image using custom linear interpolation.
    """
    toned_image = torch.empty_like(image)
    for i in range(3):
        x_vals = tonemap[i][:, 0].contiguous()
        y_vals = tonemap[i][:, 1].contiguous()
        toned_image[i] = interp(image[i], x_vals, y_vals)
    return toned_image

def colorize_tensor(value, vmin=None, vmax=None, cmap=None, colorbar=False, height=9.6, width=7.2):
    """
    Convert tensor to 3-channel RGB array using colormap (similar to plt.imshow).
    """
    assert len(value.shape) == 2
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(width, height)
    a = ax.imshow(value.detach().cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        cbar = plt.colorbar(a, fraction=0.05)
        cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.close()

    # Convert figure to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    return torch.tensor(img).permute(2, 0, 1).float()
