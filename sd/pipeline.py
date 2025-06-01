import torch
import numpy as np
from tqdm import tqdm
from ddpm  import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH//8
LATENTS_HEIGHT = HEIGHT//8

''' -----------PARAMETERS------------------------------
uncond_prompt :- negative prompt related to classifier free guidance (If nothing given generally taken as empty string "")
input_image :- input image for image to image model
strength :- if image to image model how much attention to be given to the initial (i/p) image to generate the output image... 
            more is the value of strength more noise is added to the i/p image thereby model can be more creative because it has more noise to remove thereby creating some unique image wheras if there is less noise the model can be much creative as most of the image is defined and only small amount of noise can be removed by the model so  the output will resemble more or less to  the input
do_cfg :- do classfier free guidance
cfg_scale :- the weight of how much the model should  ppay attention to the input prompt.. ranges from 1 - 14 (how much attention to be paid to conditional output wrt unconditional output where conditional output is conditioned to input prompt and unconditional output is conditioned to negative prompt(uncond_prompt))
sampler_name :- DDPM
n_inference_steps :- no of inference steps (generally 50 is considered to enough for satisfactory results)
models :- pretrained models
seed :- how  to initialise the random number generator
device :- where to create the tensors
idle_device :- when we load some model to CUDA and then don't need it then we move it to cpu
tokenizer :-
-----------PARAMETERS------------------------------'''

def generate(prompt : str, uncond_prompt: str, input_image=None, strength = 0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm",  n_inference_steps=50,
             models={}, seed=None,
             device=None,
             idle_device=None,
             tokenizer=None
            ):

        with torch.no_grad():
            #because we are inferencing the model

            if not(0< strength <=1):
                raise ValueError("strength must be between 0 and 1")
            
            if idle_device:
                to_idle = lambda x:x.to(idle_device)
            else:
                to_idle = lambda x: x

            generator = torch.Generator(device=device)  #random number generator to generate noise
            if seed is None:
                generate.seed()
            
            else:
                generator.manual_seed(seed)
                
            clip = models["clip"]                       # clip is a model taken from pretrained models
            clip.to(device)                             
            
            #### Here we are building text - to -image version
            
            if do_cfg:                                  # in classifier free guidance we infer the model twice once conditioned with input prompt and once uncond_prompt 
                # Convert the prompt into tokens using the tokenizer
                cond_tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids  # padding is set to fill the prompt to max length
                #(Batch_Size,Seq_Len)
                cond_tokens = torch.tensor(cond_tokens,dtype=torch.long,device=device)
                #(Batch_Size,Seq_Len) -> (Batch_Size,Seq_Len,Dim)
                cond_context = clip(cond_tokens)
                
                uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding="max_length",max_length=77).input_ids 
                uncond_tokens = torch.tensor(uncond_tokens,dtype = torch.long,device=device)
                #(Batch_Size,Seq_Len) -> (Batch_Size,Seq_Len,Dim)
                uncond_context = clip(uncond_tokens)
                
                # Merging conditional prompt with unconditional prompt (negative prompt is generally empty string if nothing is specified)
                # (2,Seq_Len,Dim) = (2,77,768)
                context = torch.cat([cond_context,uncond_context])      # model  will produce two outputs because model takes care of Batch_Size
                
            else:                                        # in this  there is only one pass through the UNET with the input prompt and no uncondtional input is  taken into consideration
                tokens = tokenizer.batch_encode_plus([prompt],padding="max length",max_length=77).input_ids
                tokens =   torch.tensor(tokens,dtype=torch.long,device=device)
                #(1,77,768)
                context = clip(tokens)
                
            to_idle(clip)  #  Since there is no  further use of clip we can move it to  the idle_device(CPU) because there is limited GPU memory so we offload the models back onto CPU
            
            if sampler_name == "ddpm":
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
                
            else:
                raise  ValueError(f"Unknown Sampler {sampler_name}")
            
            ##### Here we are building image to image with prompt 
            
            latents_shape = (1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
            
            if input_image:
                encoder = models["encoder"]
                encoder.to(device)
                
                input_image_tensor = input_image.resize((WIDTH,HEIGHT))
                input_image_tensor = np.array(input_image_tensor)
                
                # (Height,Width,Channel)
                input_image_tensor = torch.tensor(input_image_tensor,dtype=torch.float32)
                
                input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))                    # the values of pixel in the image are b/w 0 and 255 but UNET expects the values between -1 and 1
                
                # (Height,Width,Channel) -> (Batch_Size,Height,Width,Channel)
                input_image_tensor = input_image_tensor.unsqueeze(0)
            
                # (Batch_Size,Height,Width,Channel) -> (Batch_Size,Channel,Height,Width)
                input_image_tensor = input_image_tensor.permute(0,3,1,2)
                
                
                encoder_noise = torch.randn(latents_shape,generator=generator,device=device)
                
                # run the image through the encoder of VAE
                latents = encoder(input_image_tensor,encoder_noise)
                
                sampler.set_strength(strength=strength)
                
                # Adding noise to the latents using sampler
                latents = sampler.add_noise(latents,sampler.timesteps[0])
                
                to_idle(encoder)                            # We dont need the encoder anymore so  offloading it out of GPU to idle_device
                
            else:                                           # If no  input image is supplied so  it is a Text-To-Image model and we start with random noise in place of input_image  
                # If we are doing text-to-image start  with random  noise ~ N(0,I)
                latents = torch.randn(latents_shape,generator=generator,device=device)
                
                          
            diffusion = models["diffusion"]
            diffusion.to(device)
            
            timesteps = tqdm(sampler.timesteps)
            for i,timestep in enumerate(timesteps):                                     # for each denoising iteration UNET predicts the noise and the sampler removes it 
                #(1,320) ->  time embedding dimension to  give UNET(Diffusion model)
                time_embedding = get_time_embedding(timestep).to(device)
                
                #(Batch_Size, 4,Latents_Height,Latents_Width)
                model_input = latents
                
                if do_cfg:
                    # (Batch_Size,4,Latents_Height,Latents_Width) -> (2*Batch_Size,4,Latents_Height,Latents_Width)
                    model_input = model_input.repeat(2,1,1,1)
                    
                # model output is  the predicted noise by the UNET
                model_output = diffusion(model_input,context,time_embedding)
                
                
                if  do_cfg:
                    output_cond , output_uncond = model_output.chunk(2)  # Since for cfg double  batch size is given as input and the  model  maintains the  batch size we can chunk it in half to  get two output one with cond_input and one with uncond_input
                    model_output = cfg_scale * (output_cond - output_uncond)+output_uncond
                    
                # Remove noise predicted by the UNET done by scheduler
                latents = sampler.step(timestep,latents,model_output)
                
            to_idle(diffusion)                      #Denoising step and hence UNET work is done
            
            decoder = models["decoder"]             # To run the final latents from the diffusion model (UNET) through decoder to  get the final image
            decoder.to(device)
            
            images = decoder(latents)
            to_idle(decoder)
            
            images = rescale(images,(-1,1),(0,255),clamp=True)
            
            # (Batch_Size,Channels,Height,Width)  -> (Batch_Size,Height,Width,Channels)
            images = images.permute(0,2,3,1)          # to save the image in cpu we need the channel dimension in the last
            images = images.to("cpu",torch.uint8).numpy()
            return images[0]
        
def rescale(x,old_range,new_range,clamp=False):
    old_min,old_max = old_range
    new_min,new_max = new_range
    x-=old_min
    x*=(new_max-new_min)/(old_max-old_min)
    x+=new_min
    
    if clamp:
        x = x.clamp(new_min,new_max)
        
    return x

def get_time_embedding(timestep):            # to convert timestep into a vector of dimm 320... implemented similarly as positional embedding in Transformers
    # (160,)
    freqs = torch.pow(10000,-torch.arange(start=0,end=160,dtype=torch.float32)/160)            
    # (1,160) 
    x = torch.tensor([timestep],dtype=torch.float32)[:,None]*freqs[None]            # just like unsqueeze
    # (1,320)
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)               
                