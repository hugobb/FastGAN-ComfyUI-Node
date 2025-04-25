import torch 
import os
from fastgan import FastGAN

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
    
    def sample_latent(self, model, num_samples):
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
    
    def generate_images(self, model, latents):
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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadFastGAN": LoadFastGAN,
    "SampleLatent": SampleLatent,
    "GenerateImages": GenerateImages,
    "SaveLatent": SaveLatent,
    "LoadLatent": LoadLatent,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFastGAN": "Load FastGAN",
    "SampleLatent": "Sample Latent",
    "GenerateImages": "Generate Images",
    "SaveLatent": "Save Latent",
    "LoadLatent": "Load Latent",
}
