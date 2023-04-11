from typing import List, Tuple
from scipy import interpolate
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import abc


class GuideModel(torch.nn.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, x_img):
        pass

    @abc.abstractmethod
    def compute_loss(self, inp):
        pass


class Guider(torch.nn.Module):
    def __init__(self, sampler, guide_model, scale=1.0, verbose=False):
        """Apply classifier guidance

        Specify a guidance scale as either a scalar
        Or a schedule as a list of tuples t = 0->1 and scale, e.g.
        [(0, 10), (0.5, 20), (1, 50)]
        """
        super().__init__()
        self.sampler = sampler
        self.index = 0
        self.show = verbose
        self.guide_model = guide_model
        self.history = []

        if isinstance(scale, (Tuple, List)):
            times = np.array([x[0] for x in scale])
            values = np.array([x[1] for x in scale])
            self.scale_schedule = {"times": times, "values": values}
        else:
            self.scale_schedule = float(scale)

        self.ddim_timesteps = sampler.ddim_timesteps
        self.ddpm_num_timesteps = sampler.ddpm_num_timesteps


    def get_scales(self):
        if isinstance(self.scale_schedule, float):
            return len(self.ddim_timesteps)*[self.scale_schedule]

        interpolater = interpolate.interp1d(self.scale_schedule["times"], self.scale_schedule["values"])
        fractional_steps = np.array(self.ddim_timesteps)/self.ddpm_num_timesteps
        return interpolater(fractional_steps)

    def modify_score(self, model, e_t, x, t, c):

        # TODO look up index by t
        scale = self.get_scales()[self.index]

        if (scale == 0):
            return e_t

        sqrt_1ma = self.sampler.ddim_sqrt_one_minus_alphas[self.index].to(x.device)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            pred_x0 = model.predict_start_from_noise(x_in, t=t, noise=e_t)
            x_img = model.first_stage_model.decode((1/0.18215)*pred_x0)

            inp = self.guide_model.preprocess(x_img)
            loss = self.guide_model.compute_loss(inp)
            grads = torch.autograd.grad(loss.sum(), x_in)[0]
            correction = grads * scale

            if self.show:
                clear_output(wait=True)
                print(loss.item(), scale, correction.abs().max().item(), e_t.abs().max().item())
                self.history.append([loss.item(), scale, correction.min().item(), correction.max().item()])
                plt.imshow((inp[0].detach().permute(1,2,0).clamp(-1,1).cpu()+1)/2)
                plt.axis('off')
                plt.show()
                plt.imshow(correction[0][0].detach().cpu())
                plt.axis('off')
                plt.show()


        e_t_mod = e_t - sqrt_1ma*correction
        if self.show:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(e_t[0][0].detach().cpu(), vmin=-2, vmax=+2)
            axs[1].imshow(e_t_mod[0][0].detach().cpu(), vmin=-2, vmax=+2)
            axs[2].imshow(correction[0][0].detach().cpu(), vmin=-2, vmax=+2)
            plt.show()
        self.index += 1
        return e_t_mod