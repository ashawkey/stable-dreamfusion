import argparse
import glob
import os
from tqdm import tqdm
from collections import namedtuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from ldm.modules.evaluate.ssim import ssim


transform = transforms.Compose([transforms.ToTensor()])

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1)).view(
        in_feat.size()[0], 1, in_feat.size()[2], in_feat.size()[3]
    )
    return in_feat / (norm_factor.expand_as(in_feat) + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(
        torch.mean(
            torch.sum(in0_norm * in1_norm, dim=1).view(N, 1, X, Y), dim=2
        ).view(N, 1, 1, Y),
        dim=3,
    ).view(N)


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(
            pretrained=pretrained
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple(
            "SqueezeOutputs",
            ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"],
        )
        out = vgg_outputs(
            h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7
        )

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = models.alexnet(
            pretrained=pretrained
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = models.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = models.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = models.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = models.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = models.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple(
            "Outputs", ["relu1", "conv2", "conv3", "conv4", "conv5"]
        )
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out

# Off-the-shelf deep network
class PNet(torch.nn.Module):
    """Pre-trained network with all channels equally weighted by default"""

    def __init__(self, pnet_type="vgg", pnet_rand=False, use_gpu=True):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.Tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        self.scale = torch.Tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)

        if self.pnet_type in ["vgg", "vgg16"]:
            self.net = vgg16(pretrained=not self.pnet_rand, requires_grad=False)
        elif self.pnet_type == "alex":
            self.net = alexnet(
                pretrained=not self.pnet_rand, requires_grad=False
            )
        elif self.pnet_type[:-2] == "resnet":
            self.net = resnet(
                pretrained=not self.pnet_rand,
                requires_grad=False,
                num=int(self.pnet_type[-2:]),
            )
        elif self.pnet_type == "squeeze":
            self.net = squeezenet(
                pretrained=not self.pnet_rand, requires_grad=False
            )

        self.L = self.net.N_slices

        if use_gpu:
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if retPerLayer:
            all_scores = []
        for (kk, out0) in enumerate(outs0):
            cur_score = 1.0 - cos_sim(outs0[kk], outs1[kk])
            if kk == 0:
                val = 1.0 * cur_score
            else:
                val = val + cur_score
            if retPerLayer:
                all_scores += [cur_score]

        if retPerLayer:
            return (val, all_scores)
        else:
            return val




# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)


# The PSNR metric
def psnr(img1, img2, mask=None,reshape=False):
    b = img1.size(0)
    if not (mask is None):
        b = img1.size(0)
        mse_err = (img1 - img2).pow(2) * mask
        if reshape:
            mse_err = mse_err.reshape(b, -1).sum(dim=1) / (
                    3 * mask.reshape(b, -1).sum(dim=1).clamp(min=1)
            )
        else:
            mse_err = mse_err.view(b, -1).sum(dim=1) / (
                3 * mask.view(b, -1).sum(dim=1).clamp(min=1)
            )
    else:
        if reshape:
            mse_err = (img1 - img2).pow(2).reshape(b, -1).mean(dim=1)
        else:
            mse_err = (img1 - img2).pow(2).view(b, -1).mean(dim=1)

    psnr = 10 * (1 / mse_err).log10()
    return psnr


# The perceptual similarity metric
def perceptual_sim(img1, img2, vgg16):
    # First extract features
    dist = vgg16(img1 * 2 - 1, img2 * 2 - 1)

    return dist

def load_img(img_name, size=None):
    try:
        img = Image.open(img_name)

        if type(size) == int:
            img = img.resize((size, size))
        elif size is not None:
            img = img.resize((size[1], size[0]))

        img = transform(img).cuda()
        img = img.unsqueeze(0)
    except Exception as e:
        print("Failed at loading %s " % img_name)
        print(e)
        img = torch.zeros(1, 3, 256, 256).cuda()
        raise
    return img


def compute_perceptual_similarity(folder, pred_img, tgt_img, take_every_other):

    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    values_percsim = []
    values_ssim = []
    values_psnr = []
    folders = os.listdir(folder)
    for i, f in tqdm(enumerate(sorted(folders))):
        pred_imgs = glob.glob(folder + f + "/" + pred_img)
        tgt_imgs = glob.glob(folder + f + "/" + tgt_img)
        assert len(tgt_imgs) == 1

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])
            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        values_psnr += [psnr_sim]

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [
                min(values_percsim[2 * i], values_percsim[2 * i + 1])
            ]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    return {
        "Perceptual similarity": (avg_percsim, std_percsim),
        "PSNR": (avg_psnr, std_psnr),
        "SSIM": (avg_ssim, std_ssim),
    }


def compute_perceptual_similarity_from_list(pred_imgs_list, tgt_imgs_list,
                                            take_every_other,
                                            simple_format=True):

    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    values_percsim = []
    values_ssim = []
    values_psnr = []
    equal_count = 0
    ambig_count = 0
    for i, tgt_img in enumerate(tqdm(tgt_imgs_list)):
        pred_imgs = pred_imgs_list[i]
        tgt_imgs = [tgt_img]
        assert len(tgt_imgs) == 1

        if type(pred_imgs) != list:
            pred_imgs = [pred_imgs]

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        assert len(pred_imgs)>0
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])
            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        if psnr_sim != np.float("inf"):
            values_psnr += [psnr_sim]
        else:
            if torch.allclose(p_img, t_img):
                equal_count += 1
                print("{} equal src and wrp images.".format(equal_count))
            else:
                ambig_count += 1
                print("{} ambiguous src and wrp images.".format(ambig_count))

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [
                min(values_percsim[2 * i], values_percsim[2 * i + 1])
            ]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    if simple_format:
        # just to make yaml formatting readable
        return {
            "Perceptual similarity": [float(avg_percsim), float(std_percsim)],
            "PSNR": [float(avg_psnr), float(std_psnr)],
            "SSIM": [float(avg_ssim), float(std_ssim)],
        }
    else:
        return {
            "Perceptual similarity": (avg_percsim, std_percsim),
            "PSNR": (avg_psnr, std_psnr),
            "SSIM": (avg_ssim, std_ssim),
        }


def compute_perceptual_similarity_from_list_topk(pred_imgs_list, tgt_imgs_list,
                                                 take_every_other, resize=False):

    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    values_percsim = []
    values_ssim = []
    values_psnr = []
    individual_percsim = []
    individual_ssim = []
    individual_psnr = []
    for i, tgt_img in enumerate(tqdm(tgt_imgs_list)):
        pred_imgs = pred_imgs_list[i]
        tgt_imgs = [tgt_img]
        assert len(tgt_imgs) == 1

        if type(pred_imgs) != list:
            assert False
            pred_imgs = [pred_imgs]

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        sample_percsim = list()
        sample_ssim = list()
        sample_psnr = list()
        for p_img in pred_imgs:
            if resize:
                t_img = load_img(tgt_imgs[0], size=(256,256))
            else:
                t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])

            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
            sample_percsim.append(t_perc_sim)
            perc_sim = min(perc_sim, t_perc_sim)

            t_ssim = ssim_metric(p_img, t_img).item()
            sample_ssim.append(t_ssim)
            ssim_sim = max(ssim_sim, t_ssim)

            t_psnr = psnr(p_img, t_img).item()
            sample_psnr.append(t_psnr)
            psnr_sim = max(psnr_sim, t_psnr)

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        values_psnr += [psnr_sim]
        individual_percsim.append(sample_percsim)
        individual_ssim.append(sample_ssim)
        individual_psnr.append(sample_psnr)

    if take_every_other:
        assert False, "Do this later, after specifying topk to get proper results"
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [
                min(values_percsim[2 * i], values_percsim[2 * i + 1])
            ]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    individual_percsim = np.array(individual_percsim)
    individual_psnr = np.array(individual_psnr)
    individual_ssim = np.array(individual_ssim)

    return {
        "avg_of_best": {
            "Perceptual similarity": [float(avg_percsim), float(std_percsim)],
            "PSNR": [float(avg_psnr), float(std_psnr)],
            "SSIM": [float(avg_ssim), float(std_ssim)],
        },
        "individual": {
            "PSIM": individual_percsim,
            "PSNR": individual_psnr,
            "SSIM": individual_ssim,
        }
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="")
    args.add_argument("--pred_image", type=str, default="")
    args.add_argument("--target_image", type=str, default="")
    args.add_argument("--take_every_other", action="store_true", default=False)
    args.add_argument("--output_file", type=str, default="")

    opts = args.parse_args()

    folder = opts.folder
    pred_img = opts.pred_image
    tgt_img = opts.target_image

    results = compute_perceptual_similarity(
        folder, pred_img, tgt_img, opts.take_every_other
    )

    f = open(opts.output_file, 'w')
    for key in results:
        print("%s for %s: \n" % (key, opts.folder))
        print(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )

        f.write("%s for %s: \n" % (key, opts.folder))
        f.write(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )

    f.close()
