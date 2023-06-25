import torch 
import torch.nn as nn 
import cv2 
import numpy as np 
import os 
import time 

from dataset import get_dataloader, get_img_shape 
from ddpm import DDPM 
from network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512 
n_epochs = 100 
use_pytorch2 = True 

def train(
    ddpm: DDPM, 
    net, 
    device='cpu', 
    ckpt_path="tmp/model.pth", 
): 
    n_steps = ddpm.n_steps 
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    
    tic = time.time()
    for e in range(n_epochs): 
        total_loss = 0 
        
        for x, _ in dataloader: 
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x,t,eps)
            eps_theta = net(x_t, t.reshape(current_batch_size,1))
            loss = loss_fn(eps_theta,eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*current_batch_size
            
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f"epoch {e} loss: {total_loss} elapsed {(toc-tic):.2f}s")
       
def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=9,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        
        # imgs = einops.rearrange(imgs,
        #                         '(b1 b2) c h w -> (b1 h) (b2 w) c',
        #                         b1=int(n_sample**0.5))
        # Assuming `imgs` is a PyTorch tensor with shape (b1 * b2, c, h, w)
        b, c, h, w = imgs.shape
        b1 = int(n_sample ** 0.5)  # Calculate the value of b1
        b2 = int(b/b1)
        imgs = imgs.view(b1, b2, c, h, w)  # Reshape to (b1, b2, c, h, w)
        imgs = imgs.permute(0, 3, 1, 4, 2)  # Permute dimensions to (b1, h, b2, w, c)
        imgs = imgs.contiguous().view(b1 * h, b2 * w, c)  # Reshape to (b1 * h, b2 * w, c)

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)
        
if __name__ == "__main__": 
    os.makedirs("tmp", exist_ok=True)
        
    n_steps = 1000
    config_id = 4 
    device = "cpu"
    model_path = "tmp/model_unet_res.pth"
    
    configs = [
        convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
        unet_res_cfg
    ]
    config = configs[config_id]
    net = build_network(config, n_steps)
    if use_pytorch2:
        net = torch.compile(net) 
    ddpm = DDPM(device, n_steps)
    
    train(ddpm, net, device=device, ckpt_path=model_path)
    
    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, "assets/diffusion.jpg", device=device)