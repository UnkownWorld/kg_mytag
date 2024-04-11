import argparse
import os
from models.image_text_transformation import ImageTextTransformation
from utils.util import display_images_and_text

def write_tag(image_name,tag,folder_path):
    # 获取图片文件名（不包括扩展名）  
    image_name = os.path.splitext(os.path.basename(image_path))[0]  
    # 构建文本文件的完整路径  
    text_file_path = os.path.join(folder_path, image_name + '.txt') 
    # 将内容写入文本文件  
    with open(text_file_path, 'w', encoding='utf-8') as text_file:  
        text_file.write(tag)  
        print(f"Text file created: {text_file_path},{tag}") 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='/kaggle/working/xuejie')
    parser.add_argument('--image_src', default='examples/1.jpg')
    parser.add_argument('--out_image_name', default='output/1_result.jpg')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
    parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_b', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
    parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, <6G GPU is not recommended>')

    args = parser.parse_args()

    processor = ImageTextTransformation(args)
    for image_src in args.image_dir:        
        generated_text = processor.image_to_text(args.image_src)
        image_path = image_src
        name = os.path.splitext(os.path.basename(image_path))[0]
        dir = os.path.dirname(image_path)
        write_tag(name,generated_text,dir)
    ## then text to image
        print("*" * 50)
        print("Generated Text:")
        print(generated_text)
        print("*" * 50)

    #results = display_images_and_text(args.image_src, generated_image, generated_text, args.out_image_name)
