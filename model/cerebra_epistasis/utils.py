import os
import sys
import time
import pickle
import gzip
import torch
import contextlib
from functools import wraps, lru_cache
from filelock import FileLock

from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def uniq(arr):
    return list({el: True for el in arr}.keys())


def to_order(degree):
    return 2 * degree + 1


def map_values(fn, d):
    return {k: fn(v) for k, v in d.items()}


def safe_cat(arr, el, dim):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim=dim)


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


# 这个 batched_index_select 函数的作用是：在批次数据（batched tensor）上模拟 index_select 操作，即在指定维度 dim 上按照 indices 索引 values
def batched_index_select(values, indices, dim=1):
    # 提取 values 张量中，在指定维度 dim 之后的所有“尾部特征维度”，并将它们保存为 value_dims。
    value_dims = values.shape[(dim + 1) :]
    # 将 values 和 indices 这两个张量的 shape（张量维度信息）转换为 Python 的列表形式，并分别保存为 values_shape 和 indices_shape。
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    # 给 indices 尾部加上若干个维度（维度值为1），其个数等于 value_dims 的维度个数，从而让它能和 values 的尾部特征维对齐，以支持后续的 .expand(...) 和 torch.gather()。
    indices = indices[(..., *((None,) * len(value_dims)))]
    # 这行代码把 indices 广播成一个与 values 尾部特征维度相匹配的形状，使它可以在 gather() 操作中选择出 values 中的整块特征（比如通道、向量分量等）。
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def masked_mean(tensor, mask, dim=-1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.0)

    total_el = mask.sum(dim=dim)
    mean = tensor.sum(dim=dim) / total_el.clamp(min=1.0)
    mean.masked_fill_(total_el == 0, 0.0)
    return mean


def rand_uniform(size, min_val, max_val):
    return torch.empty(size).uniform_(min_val, max_val)


def fast_split(arr, splits, dim=0):
    axis_len = arr.shape[dim]
    # axis_len:32, splits:4
    splits = min(axis_len, max(splits, 1))

    # 整数除法（//）保证是整除结果。
    chunk_size = axis_len // splits
    # 计算在平均分割后剩下的“余数”元素个数，即最后还没被分配出去的部分。
    remainder = axis_len - chunk_size * splits
    s = 0
    for i in range(splits):
        # 是 Python 的多变量同时赋值（tuple unpacking），结合条件判断 if ... else，功能是：决定当前这一块要不要“多拿一个元素”，并更新剩余的余数。
        # if remainder > 0:
        # adjust = 1     # 当前块多分一个
        # else:
        # adjust = 0     # 均分即可
        # remainder = remainder - 1  # 减少一个余数（已经分出去了）
        adjust, remainder = 1 if remainder > 0 else 0, remainder - 1
        # torch.narrow：从张量 input 的第 dim 维上，提取从位置 start 开始、长度为 length 的子张量。
        yield torch.narrow(arr, dim, s, chunk_size + adjust)
        s += chunk_size + adjust


# 把输入 x 映射到 高维的 Fourier 特征空间
def fourier_encode(x, num_encodings=4, include_self=True, flatten=True):
    x = x.unsqueeze(-1)  # 在最后一个维度上增加一个维度
    device, dtype, orig_x = x.device, x.dtype, x
    # 对这个序列逐元素做 2 的幂运算。
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    # scales:tensor([1., 2., 4., 8.])

    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    x = rearrange(x, "b m n ... -> b m n (...)") if flatten else x
    return x


# default dtype context manager


@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)


def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        return fn(t)

    return inner


# benchmark tool


def benchmark(fn):
    def inner(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        diff = time.time() - start
        return diff, res

    return inner


# caching functions


def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner

    return cache_inner


# cache in directory


def cache_dir(dirname, maxsize=128):
    """
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    """

    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exists(dirname):
                return func(*args, **kwargs)

            os.makedirs(dirname, exist_ok=True)

            indexfile = os.path.join(dirname, "index.pkl")
            lock = FileLock(os.path.join(dirname, "mutex"))

            with lock:
                index = {}
                if os.path.exists(indexfile):
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                key = (args, frozenset(kwargs), func.__defaults__)

                if key in index:
                    filename = index[key]
                else:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            if os.path.exists(filepath):
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            print(f"compute {filename}... ", end="", flush=True)
            result = func(*args, **kwargs)
            print(f"save {filename}... ", end="", flush=True)

            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            print("done")

            return result

        return wrapper

    return decorator


def row_zscore(x, eps=1e-8):
    # x: [L,20]
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / (std + eps)


def rbf_encode(dist, num_rbf=16, rbf_min=0.0, rbf_max=20.0):
    # dist: [..., 1]
    rbf_centers = torch.linspace(rbf_min, rbf_max, num_rbf, device=dist.device)
    rbf_sigma = (rbf_max - rbf_min) / num_rbf
    return torch.exp(-((dist - rbf_centers) ** 2) / (2 * rbf_sigma**2))


def safe_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # v: [..., 3]
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def virtual_cb_from_n_ca_c(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    给 GLY 构造 virtual Cβ（坐标），常用公式：
    b = CA - N
    c = C - CA
    a = cross(b, c)
    CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
    """
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = (-0.58273431 * a) + (0.56802827 * b) + (-0.54067466 * c) + CA
    return CB


def rank_dropout(U, p: float, training: bool):
    """
    U: [..., r]
    drop whole rank channels (last dim).
    """
    if (not training) or (p <= 0.0):
        return U
    r = U.shape[-1]

    # one mask shared across all positions (most stable)
    mask = (torch.rand(r, device=U.device) > p).to(U.dtype)  # [r]
    mask = mask / (1.0 - p)  # keep expectation
    return U * mask  # broadcast on last dim
