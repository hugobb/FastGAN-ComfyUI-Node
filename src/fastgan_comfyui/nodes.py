import numpy as np
import torch 
import os
from fastgan import FastGAN
from PIL import Image
from torch.nn import functional as F


# --- Add to your FastGAN custom_nodes file ---
try:
    import lpips  # optional perceptual loss (pip install lpips)
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

# ---------- Device helper ----------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ---------- Image <-> Tensor helpers ----------
def comfy_to_model_image(img_tensor: torch.Tensor, resolution, device):
    """
    ComfyUI IMAGE: [B,H,W,C] in [0,1]
    Convert to model tensor [B,3,H,W] in [-1,1], resized to (resolution,resolution)
    """
    t = img_tensor[0]  # first in batch
    arr = (t.numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((resolution, resolution), Image.BICUBIC)
    x = np.array(pil).astype(np.float32) / 127.5 - 1.0  # [-1,1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return x

def model_to_comfy_image(x: torch.Tensor):
    """
    Model tensor [B,3,H,W] in [-1,1] -> ComfyUI IMAGE [B,H,W,C] in [0,1]
    """
    x = (x.clamp(-1, 1) + 1) / 2
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
    return torch.from_numpy(x)

# ---------- Latent helpers ----------
def randn_like_latent(latent, generator, device):
    return torch.randn_like(latent, device=device, generator=generator)

def slerp(a: torch.Tensor, b: torch.Tensor, t, eps=1e-7):
    """
    Spherical linear interpolation between two latent tensors.
    Works for shapes [B, Z] or [B, Z, 1, 1].
    """
    # flatten to [B, -1]
    a_f = a.view(a.size(0), -1)
    b_f = b.view(b.size(0), -1)
    a_norm = a_f / (a_f.norm(dim=1, keepdim=True) + eps)
    b_norm = b_f / (b_f.norm(dim=1, keepdim=True) + eps)
    dot = (a_norm * b_norm).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta) + eps
    w1 = torch.sin((1 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    out = w1 * a_f + w2 * b_f
    return out.view_as(a)

class LoadFastGAN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("FASTGAN",)
    FUNCTION = "load_model"
    CATEGORY = "FastGAN"
    
    def load_model(self, model_path):
        if not os.path.isfile(model_path):
            raise RuntimeError(f"File does not exist: {model_path}")
        model = FastGAN.load(model_path)
        return (model,)

class SampleLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FASTGAN",),
                "num_samples": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_latent"
    CATEGORY = "FastGAN"
    
    def sample_latent(self, model: FastGAN, num_samples):
        latents = model.sample_latent(num_samples)
        return (latents,)

class GenerateImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FASTGAN",),
                "latents": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_images"
    CATEGORY = "FastGAN"
    
    def generate_images(self, model: FastGAN, latents):
        images = model.generate(latents)
        images.transpose_(1, -1)
        print(images.min(), images.max())
        return (images,)

class SaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename": ("STRING", {
                    "default": "latent_output.pt"
                }),
                "save_directory": ("STRING", {
                    "default": "output/latents"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latent"
    CATEGORY = "FastGAN"

    def save_latent(self, latent, filename, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        full_path = os.path.join(save_directory, filename)
        torch.save(latent, full_path)
        print(f"[Latent Saved] → {full_path}")
        return ()

class LoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {
                    "default": "output/latents/latent_output.pt"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "FastGAN"

    def load_latent(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Latent file not found: {filepath}")
        latent = torch.load(filepath, map_location="cpu")
        return (latent,)


# =========================================
# New Node: InvertToLatent (GAN inversion)
# =========================================
class InvertToLatent:
    """
    Optimize a latent z so that G(z) reconstructs the input image.
    Uses L2 + (optional) LPIPS perceptual loss.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FASTGAN",),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "steps": ("INT", {"default": 300, "min": 10, "max": 5000}),
                "lr": ("FLOAT", {"default": 0.05, "min": 1e-4, "max": 1.0}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "use_lpips": ("BOOLEAN", {"default": True}),
                "lpips_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "device": ("STRING", {"default": str(pick_device())}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "reconstruction")
    FUNCTION = "invert"
    CATEGORY = "FastGAN"

    def invert(self, model: FastGAN, image, resolution, steps, lr, seed, use_lpips=True, lpips_weight=0.5, device="auto"):
        device = pick_device() if device == "auto" else torch.device(device)
        gen = torch.Generator(device).manual_seed(seed)

        # Expect your FastGAN exposes shape / sampler for z
        # We'll assume z shape from model.sample_latent(1)
        with torch.no_grad():
            z0 = model.sample_latent(1)  # should be a torch tensor on some device
        z = z0.detach().to(device).clone()
        z.requires_grad_(True)

        target = comfy_to_model_image(image, resolution, device)

        optim = torch.optim.Adam([z], lr=lr)
        percept = None
        if use_lpips and HAS_LPIPS:
            percept = lpips.LPIPS(net='vgg').to(device).eval()

        for _ in range(int(steps)):
            optim.zero_grad()
            x_rec = model.generate(z)  # expected [B,3,H,W] in [-1,1]
            if x_rec.dim() == 4 and x_rec.shape[1] != 3:
                # some impls return [B,H,W,C] or other, try to permute
                if x_rec.shape[-1] == 3:
                    x_rec = x_rec.permute(0, 3, 1, 2)
            # resize to match
            if x_rec.shape[-2:] != target.shape[-2:]:
                x_rec = torch.nn.functional.interpolate(x_rec, size=target.shape[-2:], mode="bilinear", align_corners=False)

            l2 = F.mse_loss(x_rec, target)

            loss = l2
            if percept is not None:
                # LPIPS expects [-1,1]
                lp = percept((x_rec).clamp(-1,1), (target).clamp(-1,1)).mean()
                loss = loss + lpips_weight * lp

            loss.backward()
            optim.step()

        # Final reconstruction
        with torch.no_grad():
            x_final = model.generate(z)
            if x_final.dim() == 4 and x_final.shape[1] != 3 and x_final.shape[-1] == 3:
                x_final = x_final.permute(0, 3, 1, 2)
            if x_final.shape[-2:] != target.shape[-2:]:
                x_final = torch.nn.functional.interpolate(x_final, size=target.shape[-2:], mode="bilinear", align_corners=False)

        return (z.detach().to("cpu"), model_to_comfy_image(x_final))

# =========================================
# New Node: GANImg2Img (invert + blend + generate)
# =========================================
class GANImg2Img:
    """
    Invert input image to latent, blend with random latent using strength, and generate.
    strength: 0.0 -> keep inverted z (preserve input)
              1.0 -> ignore input (pure random)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FASTGAN",),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "inv_steps": ("INT", {"default": 300, "min": 10, "max": 5000}),
                "seed": ("INT", {"default": 0}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "lr": ("FLOAT", {"default": 0.05, "min": 1e-4, "max": 1.0}),
                "use_lpips": ("BOOLEAN", {"default": True}),
                "lpips_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "device": ("STRING", {"default": str(pick_device())}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "run"
    CATEGORY = "FastGAN"

    def run(self, model: FastGAN, image, resolution, inv_steps, seed, strength, lr=0.05, use_lpips=True, lpips_weight=0.5, device="auto"):
        device = pick_device() if device == "auto" else torch.device(device)
        gen = torch.Generator(device).manual_seed(seed)

        # 1) Invert image -> z_hat
        inv_node = InvertToLatent()
        z_hat, recon = inv_node.invert(model, image, resolution, inv_steps, lr, seed, use_lpips, lpips_weight, str(device))

        z_hat = z_hat.to(device).detach()
        # 2) Sample random z
        with torch.no_grad():
            z_rand = model.sample_latent(1)
        z_rand = z_rand.to(device)

        # 3) Blend (SLERP) by strength
        t = torch.tensor([strength], device=device).view(-1, 1)
        z_mix = slerp(z_hat, z_rand, t)

        # 4) Generate
        with torch.no_grad():
            x = model.generate(z_mix)
            if x.dim() == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            if resolution is not None and x.shape[-2:] != (resolution, resolution):
                x = torch.nn.functional.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)

        return (model_to_comfy_image(x), z_mix.detach().to("cpu"))

class BlendLatents:
    """
    Blend (interpolate) between two LATENT tensors from FastGAN.
    strength = 0 → latent1
    strength = 1 → latent2
    mode = "slerp" (spherical) or "lerp" (linear)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "mode": ("STRING", {"default": "slerp", "choices": ["slerp", "lerp"]}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"
    CATEGORY = "FastGAN"

    def blend(self, latent1: torch.Tensor, latent2: torch.Tensor, strength, mode="slerp"):
        # Ensure same device and shape
        device = latent1.device
        latent2 = latent2.to(device)
        t = torch.tensor([strength], device=device)

        if mode == "slerp":
            z = slerp(latent1, latent2, t)
        else:
            z = (1 - strength) * latent1 + strength * latent2
        return (z,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadFastGAN": LoadFastGAN,
    "SampleLatent": SampleLatent,
    "GenerateImages": GenerateImages,
    "SaveLatent": SaveLatent,
    "LoadLatent": LoadLatent,
    "InvertToLatent": InvertToLatent,
    "GANImg2Img": GANImg2Img,
    "BlendLatents": BlendLatents,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFastGAN": "Load FastGAN",
    "SampleLatent": "Sample Latent",
    "GenerateImages": "Generate Images",
    "SaveLatent": "Save Latent",
    "LoadLatent": "Load Latent",
    "InvertToLatent": "FastGAN: Invert To Latent",
    "GANImg2Img": "FastGAN: Img2Img",
    "BlendLatents": "FastGAN: Blend Latents",
}