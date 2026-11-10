import glob, os, torch
import torch.nn.functional as F
import termcolor
from depth_anything_3.api import DepthAnything3

class MetricScaleEstimator:
    def __init__(self, model_name="depth-anything/DA3NESTED-GIANT-LARGE", device=torch.device("cuda")):
        self.model = DepthAnything3.from_pretrained(model_name).to(device)

    def estimate_scale_img(self, image):
        prediction = self.model.inference(image)
        return prediction.depth


    def get_scale_factor(self, image_name, unscaled_depth, conf_mask=None, device=torch.device("cuda")):
        metric_depth = self.estimate_scale_img([image_name])
        metric_depth = torch.from_numpy(metric_depth).to(device).float()
        unscaled_depth = torch.from_numpy(unscaled_depth).to(device).float()

        # Squeeze to 2D (H, W), handling [H, W, 1], [1, H, W], or [H, W]
        metric_depth = metric_depth.squeeze()    # -> (H, W)
        unscaled_depth = unscaled_depth.squeeze()  # -> (H, W)

        # Resize metric_depth to match unscaled_depth spatial dims
        # F.interpolate expects (N, C, H, W)
        target_h, target_w = unscaled_depth.shape
        metric_resized = F.interpolate(
            metric_depth.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()  # -> (H, W)

        if conf_mask is not None:
            unscaled_depth = unscaled_depth[conf_mask]
            metric_resized = metric_resized[conf_mask]

        scale_factor = (torch.median(metric_resized / unscaled_depth)).item()
        print(termcolor.colored(f"Estimated Metric scale factor correction {scale_factor:.4f}", "green"))
        return scale_factor

    def unload(self):
        del self.model
        torch.cuda.empty_cache()