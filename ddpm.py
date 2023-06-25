import torch 

class DDPM(): 
    def __init__(self,
                 device, 
                 n_steps: int, 
                 min_beta: float = 1e-4, 
                 max_beta: float = 2e-2, 
                 ): 
        # n_steps is T
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas 
        alpha_bars = torch.empty_like(alphas)
        product = 1 
        for i, alpha in enumerate(alphas): 
            product *= alpha 
            alpha_bars[i] = product 
        
        self.betas = betas 
        self.n_steps = n_steps 
        self.alphas = alphas 
        self.alpha_bars = alpha_bars
        
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps-1]
        alpha_prev[0] = 1 
        
        self.coef1 = torch.sqrt(alphas) * (1-alpha_prev) / (1-alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1-alpha_bars)
        
    def sample_forward(self, x, t, eps=None): 
        alpha_bar = self.alpha_bars[t].reshape(-1,1,1,1)
        if eps is None: 
            eps = torch.randn_like(x)
        res = eps*torch.sqrt(1-alpha_bar) + torch.sqrt(alpha_bar)*x 
        return res 

    def sample_backward(self,
                        img_shape, 
                        net, 
                        device, 
                        simple_var=True, 
                        clip_x0=True
                        ):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps-1,-1,-1):
            x = self.sample_backward_step(x,t,net,simple_var,clip_x0)
        return x 
    
    def sample_backward_step(self,
                             x_t, 
                             t, 
                             net, 
                             simple_var=True,
                             clip_x0=True, 
                             ):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t]*n,
                                dtype=torch.long
                                ).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)
        
        if t == 0: 
            noise = 0 
        else: 
            if simple_var: 
                var = self.betas[t]
            else: 
                var = (1-self.alpha_bars[t-1])/(
                    1-self.alpha_bars[t]
                )*self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)
        
        # eps predicted by NN is used to generate the mean 
        if clip_x0: 
            x_0 = (x_t - torch.sqrt(1-self.alpha_bars[t])*eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t]*x_0
        else: 
            mean = (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alpha_bars[t])*eps)/torch.sqrt(self.alphas[t])
        
        x_t = mean + noise 
        
        return x_t 

def visualize_forward(): 
    import cv2 
    # import einops 
    import numpy as np 
    
    from dataset import get_dataloader 
    
    n_steps = 100 
    device = 'cpu'
    batch_size = 5
    dataloader = get_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    
    ddpm = DDPM(device, n_steps)
    xts = []
    percents = torch.linspace(0,0.99,10)
    for p in percents:
        t = torch.tensor([int(n_steps*p)])
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)
    
    res = torch.stack(xts, 0)
    
    # res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    # Assuming `res` is a PyTorch tensor with shape (n1, n2, c, h, w)
    n1,n2,c,h,w = res.shape
    # After the permute() operation, the tensor has dimensions (n2, h, n1, w, c)
    # this means we arrange images in a grid 
    # 1, 3 means dim n2 h
    # 0, 4 means dim n1 w  
    res = res.permute(1, 3, 0, 4, 2)  # Swapping n1 and h, n2 and w dimensions
    res = res.reshape(n2 * h, n1 * w, c)  # Reshaping the tensor

    res = (res.clip(-1,1)+1)/2 * 255 
    res = res.cpu().numpy().astype(np.uint8) 

    cv2.imwrite('tmp/diffusion_forward.png', res)
    
    
    
if __name__ == "__main__": 
    visualize_forward()
    
        