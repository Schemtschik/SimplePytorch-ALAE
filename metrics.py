import copy
import glob
import hashlib
import html
import importlib
import io
import os
import re
import sys
import tempfile
import urllib
import urllib.request
import uuid

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.utils.data
from scipy import linalg
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    if isinstance(mu1, int): mu1 = mu1 * np.ones_like(mu2)
    if isinstance(sigma1, int): sigma1 = sigma1 * np.ones_like(sigma2)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths %s, %s' % (mu1.shape, mu2.shape)
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions %s, %s' % (sigma1.shape, sigma2.shape)

    # Implement FID distance here
    ### BEGIN SOLUTION
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    ### END SOLUTION
    return fid


class ScoreModel:
    def __init__(self, cuda=True):
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            self.dtype = torch.FloatTensor

        self.inception_model = inception_v3(pretrained=True, transform_input=True).type(self.dtype)
        self.inception_model.eval()
        # self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(self.dtype)

        # remove inception_model.fc to get pool3 output 2048 dim vector
        self.fc = self.inception_model.fc
        self.inception_model.fc = nn.Sequential()

    def __forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.inception_model(x)
        pool3_ft = x.data.cpu().numpy()  # inner activation (of the last layer)

        x = self.fc(x)  # output predictions
        return pool3_ft

    @staticmethod
    def __calc_stats(pool3_ft):
        # pool3_ft is 2048 dimensional inner activation of the InceptionV3 network
        mu = np.mean(pool3_ft, axis=0)
        sigma = np.cov(pool3_ft, rowvar=False)
        return mu, sigma

    def get_pool3_ft(self, dataloader, batch_size, n_img, model=None):
        pool3_ft = np.zeros((n_img, 2048))
        for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            batch = batch.type(self.dtype)
            if model is not None:
                batch = model(batch)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            pool3_ft[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)
        return pool3_ft

    def get_fid(self, dataset, model, batch_size=32):
        n_img = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        mu2, sigma2 = self.__calc_stats(self.get_pool3_ft(dataloader, batch_size, n_img))
        mu1, sigma1 = self.__calc_stats(self.get_pool3_ft(dataloader, batch_size, n_img, model))
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return fid


#----------------------------------------------------------------------------

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d

#----------------------------------------------------------------------------

class PPLSampler(torch.nn.Module):
    def __init__(self, model, epsilon, space, sampling, crop, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self, c):
        # Generate random latents and interpolation t-values.
        t = torch.rand([128], device=c.device) * (1 if self.sampling == 'full' else 0)
        t = torch.stack([t, t], 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.model.cfg['z_dim']], device=c.device).chunk(2)

        # Interpolate in W or Z.
        if self.space == 'w':
            w0 = self.model.F(z0)
            w1 = self.model.F(z1)
            wt0 = w0.lerp(w1, t)
            wt1 = w0.lerp(w1, t + self.epsilon)
        else: # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0 = self.model.F(zt0)
            wt1 = self.model.F(zt1)

        # Randomize noise buffers.
        # for name, buf in self.G.named_buffers():
        #     if name.endswith('.noise_const'):
        #         buf.copy_(torch.randn_like(buf))

        # Generate images.
        img = torch.cat([
            self.model.G(wt0, self.model.res_idx, 1.0),
            self.model.G(wt1, self.model.res_idx, 1.0)
        ], dim=0)
        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample to 256x256.
        # factor = self.G.img_resolution // 256
        # if factor > 1:
        #     img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)

        # Evaluate differential LPIPS.
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=True, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

#----------------------------------------------------------------------------

_feature_detector_cache = dict()


def make_cache_dir_path(*paths: str) -> str:
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)



def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True):
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

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

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

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

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)



def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]


def get_obj_from_module(module, obj_name):
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_module_from_obj_name(obj_name: str):
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_by_name(name: str):
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)

def call_func_by_name(*args, func_name: str = None, **kwargs):
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs):
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)

def compute_ppl(model, num_samples, epsilon, space, sampling, crop, batch_size, jit=False):
    vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    vgg16 = get_feature_detector(vgg16_url, num_gpus=1, rank=0, verbose=False)

    # Setup sampler.
    sampler = PPLSampler(model, epsilon=epsilon, space=space, sampling=sampling, crop=crop, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to(model.device)
    # if jit:
    #     c = torch.zeros([batch_size, opts.G.c_dim], device=model.device)
    #     sampler = torch.jit.trace(sampler, [c], check_trace=False)

    # Sampling loop.
    dist = []
    # progress = opts.progress.sub(tag='ppl sampling', num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size):
        c = torch.from_numpy(np.ones((batch_size,))).to(model.device)
        x = sampler(c)
        y = x.clone()
        dist.append(y)

    # Compute PPL.)
    dist = torch.cat(dist)[:num_samples].detach().cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)