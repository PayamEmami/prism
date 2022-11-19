import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from .utils import compute_shape, resize_image_to_vgg_input, image_to_shape, compute_shape, \
    image_to_vgg_input, vgg_input_to_image, preprocess,style_transform, denormalize, unpadding
from .learn import StyleTransfer
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision as torchvision
import torch.nn.functional as F
import math

def save_image(image, save_path):
    image = denormalize(image).mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8)
    image = image.cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)

def main():
    parser = ArgumentParser(description=("Creates artwork from content and "
                            "style image."),
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('content', metavar='<content path>',
                        help='Content image.')
    parser.add_argument('style', metavar='<style path>', help='Style image.')
    parser.add_argument('--artwork', metavar='<path>',
                        default='artwork.png',
                        help='Save artwork as <path>.')
    parser.add_argument('--init_img', metavar='<path>',
                        help=('Initialize artwork with image at <path> '
                        'instead of content image.'))
    parser.add_argument('--init_random', action='store_true',
                        help=('Initialize artwork with random image '
                        'instead of content image.'))
    parser.add_argument('--area', metavar='<int>', default=512, type=int,
                        help=("Content and style are scaled such that their "
                        "area is <int> * <int>. Artwork has the same shape "
                        "as content."))
    parser.add_argument('--iter', metavar='<int>', default=500, type=int,
                        help='Number of iterations.')
    parser.add_argument('--iter_slide_outer', metavar='<int>', default=100, type=int,
                        help='Number of iterations for outer loop of slides.')
    parser.add_argument('--iter_slide_inner', metavar='<int>', default=100, type=int,
                        help='Number of iterations for inner loop of slides.')
    parser.add_argument('--tmp_dir', metavar='<str>', default=".", type=str,
                        help='Location for saving temporary images')
    parser.add_argument('--lr', metavar='<float>', default=1, type=float,
                        help='Learning rate of the optimizer.')
    parser.add_argument('--content_weight', metavar='<int>', default=1,
                        type=int, help='Weight for content loss.')
    parser.add_argument('--style_weight', metavar='<int>', default=1000,
                        type=int, help='Weight for style loss.')
    parser.add_argument('--tv_weight', metavar='<float>', default=0.0001,
                        type=float, help='Weight for TV loss.')
    parser.add_argument('--content_weights', metavar='<str>',
                        default="{'relu_4_2':1}",
                        help=('Weights of content loss for each layer. '
                        'Put the dictionary inside quotation marks.'))
    parser.add_argument('--style_weights', metavar='<str>',
                        default=("{'relu_1_1':1,'relu_2_1':1,"
                        "'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                        help=('Weights of style loss for each layer. '
                        'Put the dictionary inside quotation marks.'))
    parser.add_argument('--avg_pool', action='store_true',
                        help='Replace max-pooling by average-pooling.')
    parser.add_argument('--no_feature_norm', action='store_false',
                        help=("Don't divide each style_weight by the square "
                        "of the number of feature maps in the corresponding "
                        "layer."))
    parser.add_argument('--preserve_color', choices=['content','style','none'],
                        default='style', help=("If 'style', change content "
                        "to match style color. If 'content', vice versa. "
                        "If 'none', don't change content or style."))
    parser.add_argument('--weights', choices=['original','normalized'],
                        default='original', help=("Weights of VGG19 Network. "
                        "Either 'original' or 'normalized' weights."))
    parser.add_argument('--device', choices=['cpu','cuda','auto'],
                        default='auto', help='Device used for training.')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision for training.')
    parser.add_argument('--use_adam', action='store_true',
                        help='Use Adam instead of LBFGS optimizer.')
    parser.add_argument('--optim_cpu', action='store_true',
                        help=('Optimize artwork on CPU to move some data from'
                        ' GPU memory to working memory.'))
    parser.add_argument('--quality', metavar='<int>', default=95, type=int,
                        help=('JPEG image quality of artwork, on a scale '
                        'from 1 to 95.'))
    parser.add_argument('--logging', metavar='<int>', default=50, type=int,
                        help=('Number of iterations between logs. '
                        'If 0, no logs.'))
    parser.add_argument('--seed', metavar='<int>', default='random',
                        help='Seed for random number generators.')
    parser.add_argument('--max-side', type=int, default=1920,
                           help='Specify the dimension of the largest size of the generated image. The aspect ratio of '
                                'the content image is preserved, which is how the size of the smaller dimension is '
                                'calculated (default 1920)')
    parser.add_argument('--pyramid', dest='pyramid', action='store_true',
                           help='Use the pyramid algorithm (on by default)')
    parser.add_argument('--no-pyramid', dest='pyramid', action='store_false',
                           help='Do not use the pyramid algorithm')
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--padding', type=int, default=32, help='padding size')
    parser.set_defaults(pyramid=False)
    parser.add_argument('--overlap_slide', dest='pyramid', action='store_true',
                           help='Use the pyramid algorithm (on by default)')
    parser.set_defaults(overlap_slide=False)
    args = parser.parse_args()

    if args.seed != 'random':
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
         torch.manual_seed(int(args.seed))

    style_transfer = StyleTransfer(lr=args.lr,
                                   content_weight=args.content_weight,
                                   style_weight=args.style_weight,
                                   tv_weight=args.tv_weight,
                                   content_weights=args.content_weights,
                                   style_weights=args.style_weights,
                                   avg_pool=args.avg_pool,
                                   feature_norm=args.no_feature_norm,
                                   weights=args.weights,
                                   preserve_color=
                                   args.preserve_color.replace('none',''),
                                   device=args.device,
                                   use_amp=args.use_amp,
                                   adam=args.use_adam,
                                   optim_cpu=args.optim_cpu,
                                   logging=args.logging)

    init_img = Image.open(args.init_img) if args.init_img else None
    with Image.open(args.content) as content, Image.open(args.style) as style:
        if args.pyramid:
           shapes = []
           init_img=None
           image_shape=(content.size[0], content.size[1], 3)
           cur_shape = compute_shape(image_shape, args.max_side)
           while max(cur_shape[0], cur_shape[1]) > 224:
            shapes = [cur_shape] + shapes
            cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2, cur_shape[2])
           do_slide=False
           for i, shape in enumerate(shapes):
            if i>0:
                init_img = Image.open(args.tmp_dir+"/"+str(i-1)+"_tmp.jpg")
            new_content = content.resize((shape[0], shape[1]))
            print('Modeling image size:', new_content.size)
            image_shape_stype=(content.size[0], content.size[1], 3)
            h, w, _ = compute_shape(image_shape_stype, int(max(shape)))
            new_style = style.resize((h, w))
            try:
                artwork = style_transfer(content, style,
                                 area=max(shape),
                                 init_random=args.init_random,
                                 init_img=init_img,
                                 iter=args.iter)
                print("modeling finished!!")
                artwork.save(args.tmp_dir+"/"+str(i)+"_tmp.jpg", quality=args.quality)
                artwork.save(args.artwork, quality=args.quality)
                artwork.close()
            except Exception as e: # work on python 3.x
                print("modeling error! Trying sliding version now!!")
                do_slide=True

                artwork = Image.open(args.artwork)
                PATCH_SIZE = args.patch_size
                PADDING = args.padding
                IMAGE_WIDTH, IMAGE_HEIGHT = content.size
                resized_init=artwork.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
                trf=style_transform()
                patches=[]
                patches_init=[]
                if args.overlap_slide==True:
                    patches = preprocess_overlap(content, padding=PADDING, transform=trf, patch_size=PATCH_SIZE, cuda=False)
                    patches_init = preprocess_overlap(resized_init, padding=PADDING, transform=trf, patch_size=PATCH_SIZE, cuda=False)
                else:
                    patches = preprocess(content, padding=PADDING, transform=trf, patch_size=PATCH_SIZE, cuda=False)
                    patches_init = preprocess(resized_init, padding=PADDING, transform=trf, patch_size=PATCH_SIZE, cuda=False)

                style_transfer = StyleTransfer(lr=args.lr,
                                       content_weight=args.content_weight,
                                       style_weight=args.style_weight,
                                               tv_weight=args.tv_weight,
                                       content_weights=args.content_weights,
                                       style_weights=args.style_weights,
                                       avg_pool=args.avg_pool,
                                       feature_norm=args.no_feature_norm,
                                       weights=args.weights,
                                       preserve_color=
                                       args.preserve_color.replace('none',''),
                                       device=args.device,
                                       use_amp=args.use_amp,
                                       adam=args.use_adam,
                                       optim_cpu=args.optim_cpu,
                                       logging=args.logging)
                stylized_patches = []
                for pch in range(patches.shape[0]):
                    image=patches[pch,:,:,:].unsqueeze(0)
                    org_shape=image.shape
                    image=denormalize(image).mul_(255.0).add_(0.5).clamp_(0, 255)
                    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8)

                    image = image.cpu().numpy()
                    image = Image.fromarray(image)
                    image.save(args.tmp_dir+"/"+"content_patch"+str(pch)+".jpg")
                    image=patches_init[pch,:,:,:].unsqueeze(0)
                    image=denormalize(image).mul_(255.0).add_(0.5).clamp_(0, 255)
                    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8)

                    image = image.cpu().numpy()
                    image = Image.fromarray(image)
                    image.save(args.tmp_dir+"/"+"init_patch"+str(pch)+".jpg")
                    init_image = Image.open(args.tmp_dir+"/"+"init_patch"+str(pch)+".jpg")
                    content_image = Image.open(args.tmp_dir+"/"+"content_patch"+str(pch)+".jpg")
                    stil_new=style.resize((org_shape[2],org_shape[3]))
                    init_image.close()
                    content_image.close()
                for it in range(args.iter_slide_outer):
                    for pch in range(patches.shape[0]):
                        content_image = Image.open(args.tmp_dir+"/"+"content_patch"+str(pch)+".jpg")
                        init_image = Image.open(args.tmp_dir+"/"+"init_patch"+str(pch)+".jpg")
                        artwork2 = style_transfer(content_image, style,
                                         area=max(org_shape),
                                         init_random=False,
                                         init_img=init_image,
                                         iter=args.iter_slide_inner,fix=False)
                        content_image.close()
                        init_image.close()
                        #artwork2.save("content_patch"+str(pch)+".jpg", quality=100)
                        stylized_patch = trf(artwork2).unsqueeze(0).to(args.device)
                        stylized_patch = F.interpolate(stylized_patch, org_shape[2:], mode='bilinear', align_corners=True)
                        save_image(stylized_patch, args.tmp_dir+"/"+"content_patch"+str(pch)+".jpg")
                for pch in range(patches.shape[0]):
                    content_image = Image.open(args.tmp_dir+"/"+"content_patch"+str(pch)+".jpg")
                    stylized_patch = trf(content_image).unsqueeze(0).to(args.device)
                    stylized_patch = unpadding(stylized_patch, padding=PADDING)
                    stylized_patches.append(stylized_patch.cpu())
                    content_image.close()

                stylized_patches = torch.cat(stylized_patches, dim=0)
                b, c, h, w = stylized_patches.shape
                stylized_patches = stylized_patches.unsqueeze(dim=0)
                stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
                output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
                stylized_image = F.fold(stylized_patches, output_size=output_size,
                                kernel_size=(h, w), stride=(h, w))
                save_image(stylized_image, args.artwork)
                break


        else:
            artwork = style_transfer(content, style,
                                 area=args.area,
                                 init_random=args.init_random,
                                 init_img=init_img,
                                 iter=args.iter)
            artwork.save(args.artwork, quality=args.quality)
            artwork.close()
    if init_img:
        init_img.close()

if __name__ == '__main__':
    main()
