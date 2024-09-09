import torch

def get_distance_from_tensor(tensor):
    h, w = tensor.shape
    center_x, center_y = h // 2, w // 2
    patch = tensor[center_x-1:center_x+2, center_y-1:center_y+2]
    patch_mean = patch.mean()
    return patch_mean



def ellipse_area(a, b):
    return torch.pi * a * b

def calculate_ellipse_pixels(image_height, image_width):
    a = image_width / 2
    b = image_height / 2
    
    area = ellipse_area(torch.tensor(a), torch.tensor(b))
    
    center_x, center_y = image_width // 2, image_height // 2
    
    mask = torch.zeros((image_height, image_width), dtype=torch.float32)
    
    y, x = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing='ij')
    
    ellipse_equation = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2)
    
    mask[ellipse_equation <= 1] = 1.0
    
    ellipse_pixels = mask.sum().item()
    
    return area.item(), ellipse_pixels
