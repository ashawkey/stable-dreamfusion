import time
import torch
from diffusers import UNet2DConditionModel
import functools
from dataclasses import dataclass

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
runs_per_experiment = 50

# example inputs
sample = torch.randn(2, 4, 64, 64).cuda()
timestep = torch.rand(1).cuda() * 999
encoder_hidden_states = torch.randn(2, 77, 768).cuda()
inputs = (sample, timestep, encoder_hidden_states)

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", use_auth_token=True).to("cuda")
unet.eval()
unet.to(memory_format=torch.channels_last)  # use channels_last memory format
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
     with torch.inference_mode():
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")

# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        #inputs = generate_inputs()
        orig_output = unet_traced(*inputs)

# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class TracedUNet(torch.nn.Module):
    def __init__(self, device, token):
        super().__init__()
        self.device = device
        self.token = token
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", use_auth_token=self.token).to(self.device)
        self.in_channels = self.unet.in_channels
        self.unet_traced = torch.jit.load("unet_traced.pt")

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = self.unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)
    
traced_loaded_unet = TracedUNet("cuda", True).to("cuda")

# benchmarking
with torch.inference_mode():
    for _ in range(2*n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(runs_per_experiment):
            orig_output = traced_loaded_unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet loaded traced inference took {time.time() - start_time:.2f} seconds")
