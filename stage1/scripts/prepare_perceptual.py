"""Download the VGG16 weights used by LPIPS (and, with --fid, the pytorch-fid Inception weights)."""
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fid', action='store_true', help='Also cache official pytorch-fid Inception weights')
args = parser.parse_args()
from torchvision.models import vgg16, VGG16_Weights
vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
if args.fid:
    from pytorch_fid.inception import InceptionV3
    InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
print('Cached the auxiliary weights.')
