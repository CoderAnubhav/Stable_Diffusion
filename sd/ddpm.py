import torch
import numpy as np

class DDPMSampler:
    
    def __init__(self,generator:torch.Generator,num_training_steps=1000,beta_start: float = 0.00085,beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start**0.5,beta_end**0.5,num_training_steps,dtype=torch.float32)**2    #this  is  from diffusers library in huggingface and is called the  scaled linear schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())                 # 1000 -> 0 this is incase we need to make 1000 timesteps as is done in case of training
                                                                                                        #In Python, the [::-1] slice notation works for any 1D array (like a NumPy array or a Python list). It means:
                                                                                                        #start:stop:step
                                                                                                        #Leaving start and stop empty means "from start to end"
                                                                                                        #step = -1 means "step backwards"
                                                                                                        #So, for a 1D array, [::-1] returns a new array with the elements in reverse order.

        
    
    def set_inference_timesteps(self,num_inference_steps=50):
        """
        Set the number of inference steps for the DDPM sampler.
        """
        self.num_inference_steps = num_inference_steps
        # 999, 998,  997  ... 0 = 1000 steps
        # 999, 999-20, 999-40, ...,0 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
    def _get_previous_timestep(self,timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t
    
    
    def _get_variance(self,timestep: int)-> torch.Tensor:
        
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one
        current_beta_t = 1- alpha_prod_t / alpha_prod_t_prev
        
        # computed using eq(7) of  DDPM paper
        variance = (1 - alpha_prod_t_prev)/(1-alpha_prod_t) * current_beta_t 
        # torch.clamp() means that all values in the variance tensor are limited to be at least 1e-20.
        # If any value in variance is less than 1e-20, it will be set to 1e-20.
        # This prevents the variance from becoming zero or negative, which could cause numerical instability or errors (like division by zero) in later calculations.
        variance = torch.clamp(variance , min=1e-20)
        return variance    
    
    
    def set_strength(self,strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)        # This implies if strength = 0.8 then 1-0.8 =0.2 that is 20% of timesteps  will be skipped
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step  
    
    def step(self, timestep:int, latents: torch.Tensor, model_output:torch.Tensor):           # To denoise a noisy sample
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1- alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1- current_alpha_t
        
        # Compute the predicted original sample  using formula (15) of the DDPM paper
        pred_original_sample = (latents- beta_prod_t ** (0.5) *model_output) / alpha_prod_t ** (0.5)
        
        # Compute the coefficient for pred_original_sample and current_sample x_t in eq(7)
        pred_original_sample_coeff = (alpha_prod_t_prev** (0.5) * current_beta_t)/beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        
        # Compute the predicted previous sample mean eq(7)
        #pred_prev_sample = pred_original_sample * pred_original_sample + current_sample_coeff * latents
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        if t>0:
            device = model_output.device
            noise = torch.randn(model_output.shape,generator = self.generator,device=device,dtype = model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise         # Std Dev
        
        # N (0,1) --> N(mu,sigma^2)
        # X = mu + sigma * Z where Z ~ N(0,1)
        
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
            
        
    
    def add_noise(self,original_samples: torch.FloatTensor,timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device=original_samples.device,dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alpha_cumprod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape)<len(original_samples.shape):           # keep adding dimensions to sum and multiply them together
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = (1-alpha_cumprod[timesteps])**0.5  # Standard Deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        # According  to  equation (4) of the DDPM paper
        # Z ~ N(0,1) -> N(mean,variance)
        # X = mean + stdev * Z
        noise = torch.randn(original_samples.shape,generator=self.generator,device=original_samples.device,dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) *  noise
        return noisy_samples
    
    
    
    
 