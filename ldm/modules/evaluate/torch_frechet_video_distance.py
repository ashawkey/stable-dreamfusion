# based on https://github.com/universome/fvd-comparison/blob/master/compare_models.py; huge thanks!
import os
import numpy as np
import io
import re
import requests
import html
import hashlib
import urllib
import urllib.request
import scipy.linalg
import multiprocessing as mp
import glob


from tqdm import tqdm
from typing import Any, List, Tuple, Union, Dict, Callable

from torchvision.io import read_video
import torch; torch.set_grad_enabled(False)
from einops import rearrange

from nitro.util import isvideo

def compute_frechet_distance(mu_sample,sigma_sample,mu_ref,sigma_ref) -> float:
    print('Calculate frechet distance...')
    m = np.square(mu_sample - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_sample, sigma_ref), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_sample + sigma_ref - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma


def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

def load_video(ip):
    vid, *_ = read_video(ip)
    vid = rearrange(vid, 't h w c -> t c h w').to(torch.uint8)
    return vid

def get_data_from_str(input_str,nprc = None):
    assert os.path.isdir(input_str), f'Specified input folder "{input_str}" is not a directory'
    vid_filelist = glob.glob(os.path.join(input_str,'*.mp4'))
    print(f'Found {len(vid_filelist)} videos in dir {input_str}')

    if nprc is None:
        try:
            nprc = mp.cpu_count()
        except NotImplementedError:
            print('WARNING: cpu_count() not avlailable, using only 1 cpu for video loading')
            nprc = 1

    pool = mp.Pool(processes=nprc)

    vids = []
    for v in tqdm(pool.imap_unordered(load_video,vid_filelist),total=len(vid_filelist),desc='Loading videos...'):
        vids.append(v)


    vids = torch.stack(vids,dim=0).float()

    return vids

def get_stats(stats):
    assert os.path.isfile(stats) and stats.endswith('.npz'), f'no stats found under {stats}'

    print(f'Using precomputed statistics under {stats}')
    stats = np.load(stats)
    stats = {key: stats[key] for key in stats.files}

    return stats




@torch.no_grad()
def compute_fvd(ref_input, sample_input, bs=32,
                ref_stats=None,
                sample_stats=None,
                nprc_load=None):



    calc_stats = ref_stats is None or sample_stats is None

    if calc_stats:

        only_ref = sample_stats is not None
        only_sample = ref_stats is not None


        if isinstance(ref_input,str) and not only_sample:
            ref_input = get_data_from_str(ref_input,nprc_load)

        if isinstance(sample_input, str) and not only_ref:
            sample_input = get_data_from_str(sample_input, nprc_load)

        stats = compute_statistics(sample_input,ref_input,
                                        device='cuda' if torch.cuda.is_available() else 'cpu',
                                        bs=bs,
                                        only_ref=only_ref,
                                        only_sample=only_sample)

        if only_ref:
            stats.update(get_stats(sample_stats))
        elif only_sample:
            stats.update(get_stats(ref_stats))



    else:
        stats = get_stats(sample_stats)
        stats.update(get_stats(ref_stats))

    fvd = compute_frechet_distance(**stats)

    return {'FVD' : fvd,}


@torch.no_grad()
def compute_statistics(videos_fake, videos_real, device: str='cuda', bs=32, only_ref=False,only_sample=False) -> Dict:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)



    assert not (only_sample and only_ref), 'only_ref and only_sample arguments are mutually exclusive'

    ref_embed, sample_embed = [], []

    info = f'Computing I3D activations for FVD score with batch size {bs}'

    if only_ref:

        if not isvideo(videos_real):
            # if not is video we assume to have numpy arrays pf shape (n_vids, t, h, w, c) in range [0,255]
            videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).float()
            print(videos_real.shape)

        if videos_real.shape[0] % bs == 0:
            n_secs = videos_real.shape[0] // bs
        else:
            n_secs = videos_real.shape[0] // bs + 1

        videos_real = torch.tensor_split(videos_real, n_secs, dim=0)

        for ref_v in tqdm(videos_real, total=len(videos_real),desc=info):

            feats_ref = detector(ref_v.to(device).contiguous(), **detector_kwargs).cpu().numpy()
            ref_embed.append(feats_ref)

    elif only_sample:

        if not isvideo(videos_fake):
            # if not is video we assume to have numpy arrays pf shape (n_vids, t, h, w, c) in range [0,255]
            videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).float()
            print(videos_fake.shape)

        if videos_fake.shape[0] % bs == 0:
            n_secs = videos_fake.shape[0] // bs
        else:
            n_secs = videos_fake.shape[0] // bs + 1

        videos_real = torch.tensor_split(videos_real, n_secs, dim=0)

        for sample_v in tqdm(videos_fake, total=len(videos_real),desc=info):
            feats_sample = detector(sample_v.to(device).contiguous(), **detector_kwargs).cpu().numpy()
            sample_embed.append(feats_sample)


    else:

        if not isvideo(videos_real):
            # if not is video we assume to have numpy arrays pf shape (n_vids, t, h, w, c) in range [0,255]
            videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).float()

        if not isvideo(videos_fake):
            videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).float()

        if videos_fake.shape[0] % bs == 0:
            n_secs = videos_fake.shape[0] // bs
        else:
            n_secs = videos_fake.shape[0] // bs + 1

        videos_real = torch.tensor_split(videos_real, n_secs, dim=0)
        videos_fake = torch.tensor_split(videos_fake, n_secs, dim=0)

        for ref_v, sample_v in tqdm(zip(videos_real,videos_fake),total=len(videos_fake),desc=info):
            # print(ref_v.shape)
            # ref_v = torch.nn.functional.interpolate(ref_v, size=(sample_v.shape[2], 256, 256), mode='trilinear', align_corners=False)
            # sample_v = torch.nn.functional.interpolate(sample_v, size=(sample_v.shape[2], 256, 256), mode='trilinear', align_corners=False)


            feats_sample = detector(sample_v.to(device).contiguous(), **detector_kwargs).cpu().numpy()
            feats_ref = detector(ref_v.to(device).contiguous(), **detector_kwargs).cpu().numpy()
            sample_embed.append(feats_sample)
            ref_embed.append(feats_ref)

    out = dict()
    if len(sample_embed) > 0:
        sample_embed = np.concatenate(sample_embed,axis=0)
        mu_sample, sigma_sample = compute_stats(sample_embed)
        out.update({'mu_sample': mu_sample,
                    'sigma_sample': sigma_sample})

    if len(ref_embed) > 0:
        ref_embed = np.concatenate(ref_embed,axis=0)
        mu_ref, sigma_ref = compute_stats(ref_embed)
        out.update({'mu_ref': mu_ref,
                    'sigma_ref': sigma_ref})


    return out
