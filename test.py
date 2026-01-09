import torch
import torch.nn.functional as F
import sys
import numpy as np
import os
import time
import argparse
import cv2
import matplotlib.pyplot as plt
from models.model import Network
from data import test_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=int, default=1, help='select gpu id')
parser.add_argument('--test_path', type=str, default='E://Newfolder//Exp//ors-4199//Test//', help='test dataset path')
parser.add_argument('--save_path', type=str, default='E://Models//Our//384/results', help='path to save test results')
parser.add_argument('--save_attention', action='store_true', default=True, help='save attention visualizations')
parser.add_argument('--attention_interval', type=int, default=50, help='save attention every N images')
opt = parser.parse_args()

dataset_path = opt.test_path

# Set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
print(f'USE GPU {opt.gpu_id}')

# Load the model
model = Network()
model_path = 'E://Models//Our//384//384Net_epoch_best.pth'
model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
model.cuda()
model.eval()

# Test on a single dataset
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

image_root = opt.test_path
gt_root = 'E://Newfolder//Exp//ors-4199//GT//Test//'

test_loader = test_dataset(image_root, gt_root, opt.testsize)
time_sum = 0.0

def save_attention_visualizations(model, image, save_path, name):
    """Save attention heatmaps for a given image"""
    with torch.no_grad():
        # Run forward to populate attention weights
        model(image)
        
        # Get attention visualizations
        attention_viz = model.get_attention_visualization(batch_idx=0)
        
        # Create attention visualization directory
        attention_dir = os.path.join(save_path, 'attention_heatmaps')
        if not os.path.exists(attention_dir):
            os.makedirs(attention_dir)
        
        # Save each heatmap
        for viz_name, heatmap in attention_viz.items():
            plt.figure(figsize=(12, 10))
            
            # Create subplot with original image and attention heatmap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Original image
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            ax1.imshow(img_np)
            ax1.set_title('Input Image')
            ax1.axis('off')
            
            # Attention heatmap
            im = ax2.imshow(heatmap, cmap='viridis')
            ax2.set_title(f'Attention: {viz_name}')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)
            
            # Save figure
            viz_path = os.path.join(attention_dir, f'{name}_{viz_name}.png')
            plt.savefig(viz_path, bbox_inches='tight', dpi=150)
            plt.close()

def save_quantitative_analysis(model, save_path, test_loader_size):
    """Save quantitative attention analysis"""
    analysis = model.get_quantitative_analysis()
    
    if analysis:
        analysis_path = os.path.join(save_path, 'attention_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write("DA-STF Attention Quantitative Analysis\n")
            f.write("=" * 60 + "\n")
            f.write(f"Tested on {test_loader_size} images\n\n")
            
            # Summary first
            f.write("SUMMARY METRICS:\n")
            f.write("-" * 40 + "\n")
            if 'summary' in analysis:
                for key, value in analysis['summary'].items():
                    f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Detailed analysis
            for category, metrics in analysis.items():
                if category == 'summary':
                    continue
                    
                f.write(f"{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 40 + "\n")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # Interpretation
            f.write("INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            if 'summary' in analysis:
                summary = analysis['summary']
                f.write(f"- Average Entropy: {summary.get('avg_entropy', 0):.4f} ")
                if summary.get('avg_entropy', 0) < 2.0:
                    f.write("(Focused attention)\n")
                else:
                    f.write("(Dispersed attention)\n")
                    
                f.write(f"- Average Sparsity: {summary.get('avg_sparsity', 0):.4f} ")
                if summary.get('avg_sparsity', 0) > 0.7:
                    f.write("(Sparse attention pattern)\n")
                else:
                    f.write("(Dense attention pattern)\n")
                    
                f.write(f"- Average Diversity: {summary.get('avg_diversity', 0):.4f} ")
                if summary.get('avg_diversity', 0) > 0.1:
                    f.write("(Diverse attention heads)\n")
                else:
                    f.write("(Similar attention heads)\n")

# Main test loop
for i in range(test_loader.size):
    image, gt, name, image_for_post = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    
    # Generate and save attention visualizations
    if opt.save_attention and i % opt.attention_interval == 0:
        print(f'Generating attention visualizations for {name}')
        save_attention_visualizations(model, image, save_path, name.split('.')[0])
        model.reset_attention_analysis()  # Reset for next image
    
    # Measure inference time
    start_time = time.time()
    res = model(image)
    end_time = time.time()
    time_sum += (end_time - start_time)
    
    # Process results
    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    save_image_path = os.path.join(save_path, name)
    print('save img to:', save_image_path)
    cv2.imwrite(save_image_path, res * 255)

# Save timing results
avg_time = time_sum / test_loader.size
fps = test_loader.size / time_sum
time_save_path = os.path.join(save_path, 'inference_time.txt')
with open(time_save_path, 'w') as f:
    f.write(f"Tested on {test_loader.size} images\n")
    f.write(f"Total inference time: {time_sum:.2f} seconds\n")
    f.write(f"Average running time per image: {avg_time:.5f} seconds\n")
    f.write(f"Average FPS: {fps:.2f}\n")

# Save quantitative attention analysis
save_quantitative_analysis(model, save_path, test_loader.size)

print('Test Done!')
print(f'Metrics saved to {time_save_path}')
print(f'Attention analysis saved to {os.path.join(save_path, "attention_analysis.txt")}')