import time
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
import functools

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50



pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
)#.to("cuda")
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing()
pipe.unet.to(memory_format=torch.channels_last)
pipe.enable_attention_slicing(1)
#if is_xformers_available(): # xformers seem to use more vram...
#    pipe.enable_xformers_memory_efficient_attention()
unet = pipe.unet
unet.eval()
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# load inputs
train_latent_model_input = torch.load("train_latent_model_input.pt").to(torch.float16)
train_t = torch.load("train_t.pt").to(torch.float16)
train_text_embeddings = torch.load("train_text_embeddings.pt").to(torch.float16)

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = (train_latent_model_input, train_t, train_text_embeddings)
        orig_output = unet(*inputs)


# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        # Should have been the produce_latents inputs, but I can't seem to get a hold of them
        inputs = (train_latent_model_input, train_t, train_text_embeddings)
        orig_output = unet_traced(*inputs)


# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")
