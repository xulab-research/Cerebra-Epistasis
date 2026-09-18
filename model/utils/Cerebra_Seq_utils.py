# Shared inference and training helpers; no local Cerebra_Seq package is required.
# Extracted from the project's OpenFold/AlphaFold-derived helpers.
# Arithmetic and PDB formatting are intentionally preserved.
# See the retained Apache-2.0 notices below.

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import collections
import functools
import dataclasses
import io
import string
import numpy as np
import torch
import torch.nn as nn

# Public helpers and residue constants; imported libraries remain private.
__all__ = [
    "rot_matmul",
    "rot_vec_mul",
    "identity_rot_mats",
    "identity_trans",
    "identity_quats",
    "quat_to_rot",
    "quat_multiply",
    "quat_multiply_by_vec",
    "invert_rot_mat",
    "invert_quat",
    "Rotation",
    "Rigid",
    "batched_gather",
    "Protein",
    "get_pdb_headers",
    "to_pdb",
    "from_prediction",
    "torsion_angles_to_frames",
    "frames_and_literature_positions_to_atom14_pos",
    "hu_model_pred_to_atom14_pos",
    "make_atom14_masks",
    "cerebra_autocast",
    "clear_cuda_cache",
    "ensure_model_on_device",
    "offload_model_to_cpu",
    "offload_esm_models",
    "should_offload_esm",
    "read_fasta_sequence",
    "load_cerebra_model",
    "build_cerebra_batch",
    "prepare_cerebra_inputs",
    "sequence_to_hhblits_ids",
    "select_anchor_indices",
    "move_batch_to_model",
    "NormQuaternion",
    "QuaternionMM",
    "NormQuaternionMM",
    "Rotation2Quaternion",
    "NormVec",
    "PsiPhi",
    "comp_label",
    "tensor_to_numpy",
    "AnchorFrameConsensus",
    "reduce_plddt_output",
    "parse_fasta_file",
    "collect_fasta_files",
    "write_feature_pt",
    "get_relax_backend",
    "check_relax_environment",
    "relax_prediction",
    "feature_output_complete",
    "rc",
    "HHBLITS_AA_TO_ID",
]




# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




def rot_matmul(
    a: torch.Tensor, 
    b: torch.Tensor
) -> torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0), 
            row_mul(1), 
            row_mul(2),
        ], 
        dim=-2
    )


def rot_vec_mul(
    r: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None, 
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(
        3, dtype=dtype, device=device, requires_grad=requires_grad
    )
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3), 
        dtype=dtype, 
        device=device, 
        requires_grad=requires_grad
    )
    return trans


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros(
        (*batch_dims, 4), 
        dtype=dtype, 
        device=device, 
        requires_grad=requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.

        Args:
            quat: [*, 4] quaternions
        Returns:
            [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def _rotation_matrix_to_quaternion(
    rot: torch.Tensor,
):
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

_QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC
}

@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat *
        quat1[..., :, None, None] *
        quat2[..., None, :, None],
        dim=(-3, -2)
      )


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat *
        quat[..., :, None, None] *
        vec[..., None, :, None],
        dim=(-3, -2)
    )


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat ** 2, dim=-1, keepdim=True)
    return inv


class Rotation:
    """
        A 3D rotation. Depending on how the object is initialized, the
        rotation is represented by either a rotation matrix or a
        quaternion, though both formats are made available by helper functions.
        To simplify gradient computation, the underlying format of the
        rotation cannot be changed in-place. Like Rigid, the class is designed
        to mimic the behavior of a torch Tensor, almost as if each Rotation
        object were a tensor of rotations, in one format or another.
    """
    def __init__(self,
        rot_mats: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        normalize_quats: bool = True,
    ):
        """
            Args:
                rot_mats:
                    A [*, 3, 3] rotation matrix tensor. Mutually exclusive with
                    quats
                quats:
                    A [*, 4] quaternion. Mutually exclusive with rot_mats. If
                    normalize_quats is not True, must be a unit quaternion
                normalize_quats:
                    If quats is specified, whether to normalize quats
        """
        if((rot_mats is None and quats is None) or 
            (rot_mats is not None and quats is not None)):
            raise ValueError("Exactly one input argument must be specified")

        if((rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or 
            (quats is not None and quats.shape[-1] != 4)):
            raise ValueError(
                "Incorrectly shaped rotation matrix or quaternion"
            )

        # Force full-precision
        if(quats is not None):
            quats = quats.to(dtype=torch.float32)
        if(rot_mats is not None):
            rot_mats = rot_mats.to(dtype=torch.float32)

        if(quats is not None and normalize_quats):
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rotation:
        """
            Returns an identity Rotation.

            Args:
                shape:
                    The "shape" of the resulting Rotation object. See documentation
                    for the shape property
                dtype:
                    The torch dtype for the rotation
                device:
                    The torch device for the new rotation
                requires_grad:
                    Whether the underlying tensors in the new rotation object
                    should require gradient computation
                fmt:
                    One of "quat" or "rot_mat". Determines the underlying format
                    of the new object's rotation 
            Returns:
                A new identity rotation
        """
        if(fmt == "rot_mat"):
            rot_mats = identity_rot_mats(
                shape, dtype, device, requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(fmt == "quat"):
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods

    def __getitem__(self, index: Any) -> Rotation:
        """
            Allows torch-style indexing over the virtual shape of the rotation
            object. See documentation for the shape property.

            Args:
                index:
                    A torch index. E.g. (1, 3, 2), or (slice(None,))
            Returns:
                The indexed rotation
        """
        if type(index) != tuple:
            index = (index,)

        if(self._rot_mats is not None):
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif(self._quats is not None):
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(self,
        right: torch.Tensor,
    ) -> Rotation:
        """
            Pointwise left multiplication of the rotation with a tensor. Can be
            used to e.g. mask the Rotation.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if(self._rot_mats is not None):
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(self._quats is not None):
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self,
        left: torch.Tensor,
    ) -> Rotation:
        """
            Reverse pointwise multiplication of the rotation with a tensor.

            Args:
                left:
                    The left multiplicand
            Returns:
                The product
        """
        return self.__mul__(left)
    
    # Properties

    @property
    def shape(self) -> torch.Size:
        """
            Returns the virtual shape of the rotation object. This shape is
            defined as the batch dimensions of the underlying rotation matrix
            or quaternion. If the Rotation was initialized with a [10, 3, 3]
            rotation matrix tensor, for example, the resulting shape would be
            [10].
        
            Returns:
                The virtual shape of the rotation object
        """
        s = None
        if(self._quats is not None):
            s = self._quats.shape[:-1]
        else:
            s = self._rot_mats.shape[:-2]

        return s

    @property
    def dtype(self) -> torch.dtype:
        """
            Returns the dtype of the underlying rotation.

            Returns:
                The dtype of the underlying rotation
        """
        if(self._rot_mats is not None):
            return self._rot_mats.dtype
        elif(self._quats is not None):
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """
            The device of the underlying rotation

            Returns:
                The device of the underlying rotation
        """
        if(self._rot_mats is not None):
            return self._rot_mats.device
        elif(self._quats is not None):
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """
            Returns the requires_grad property of the underlying rotation

            Returns:
                The requires_grad property of the underlying tensor
        """
        if(self._rot_mats is not None):
            return self._rot_mats.requires_grad
        elif(self._quats is not None):
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def get_rot_mats(self) -> torch.Tensor:
        """
            Returns the underlying rotation as a rotation matrix tensor.

            Returns:
                The rotation as a rotation matrix tensor
        """
        rot_mats = self._rot_mats
        if(rot_mats is None):
            if(self._quats is None):
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quat_to_rot(self._quats)

        return rot_mats 

    def get_quats(self) -> torch.Tensor:
        """
            Returns the underlying rotation as a quaternion tensor.

            Depending on whether the Rotation was initialized with a
            quaternion, this function may call torch.linalg.eigh.

            Returns:
                The rotation as a quaternion tensor.
        """
        quats = self._quats
        if(quats is None):
            if(self._rot_mats is None):
                raise ValueError("Both rotations are None")
            else:
                quats = _rotation_matrix_to_quaternion(self._rot_mats)

        return quats

    def get_cur_rot(self) -> torch.Tensor:
        """
            Return the underlying rotation in its current form

            Returns:
                The stored rotation
        """
        if(self._rot_mats is not None):
            return self._rot_mats
        elif(self._quats is not None):
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    # Rotation functions

    def compose_q_update_vec(self, 
        q_update_vec: torch.Tensor, 
        normalize_quats: bool = True
    ) -> Rotation:
        """
            Returns a new quaternion Rotation after updating the current
            object's underlying rotation with a quaternion update, formatted
            as a [*, 3] tensor whose final three columns represent x, y, z such 
            that (1, x, y, z) is the desired (not necessarily unit) quaternion
            update.

            Args:
                q_update_vec:
                    A [*, 3] quaternion update tensor
                normalize_quats:
                    Whether to normalize the output quaternion
            Returns:
                An updated Rotation
        """
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(
            rot_mats=None, 
            quats=new_quats, 
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: Rotation) -> Rotation:
        """
            Compose the rotation matrices of the current Rotation object with
            those of another.

            Args:
                r:
                    An update rotation object
            Returns:
                An updated rotation object
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool = True) -> Rotation:
        """
            Compose the quaternions of the current Rotation object with those
            of another.

            Depending on whether either Rotation was initialized with
            quaternions, this function may call torch.linalg.eigh.

            Args:
                r:
                    An update rotation object
            Returns:
                An updated rotation object
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(
            rot_mats=None, quats=new_quats, normalize_quats=normalize_quats
        )

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
            Apply the current Rotation as a rotation matrix to a set of 3D
            coordinates.

            Args:
                pts:
                    A [*, 3] set of points
            Returns:
                [*, 3] rotated points
        """
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
            The inverse of the apply() method.

            Args:
                pts:
                    A [*, 3] set of points
            Returns:
                [*, 3] inverse-rotated points
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats) 
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self) -> Rotation:
        """
            Returns the inverse of the current Rotation.

            Returns:
                The inverse of the current Rotation
        """
        if(self._rot_mats is not None):
            return Rotation(
                rot_mats=invert_rot_mat(self._rot_mats), 
                quats=None
            )
        elif(self._quats is not None):
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(self, 
        dim: int,
    ) -> Rigid:
        """
            Analogous to torch.unsqueeze. The dimension is relative to the
            shape of the Rotation object.
            
            Args:
                dim: A positive or negative dimension index.
            Returns:
                The unsqueezed Rotation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if(self._rot_mats is not None):
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(self._quats is not None):
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(
        rs: Sequence[Rotation], 
        dim: int,
    ) -> Rigid:
        """
            Concatenates rotations along one of the batch dimensions. Analogous
            to torch.cat().

            Note that the output of this operation is always a rotation matrix,
            regardless of the format of input rotations.

            Args:
                rs: 
                    A list of rotation objects
                dim: 
                    The dimension along which the rotations should be 
                    concatenated
            Returns:
                A concatenated Rotation object in rotation matrix format
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None) 

    def map_tensor_fn(self, 
        fn: Callable[torch.Tensor, torch.Tensor]
    ) -> Rotation:
        """
            Apply a Tensor -> Tensor function to underlying rotation tensors,
            mapping over the rotation dimension(s). Can be used e.g. to sum out
            a one-hot batch dimension.

            Args:
                fn:
                    A Tensor -> Tensor function to be mapped over the Rotation 
            Returns:
                The transformed Rotation object
        """ 
        if(self._rot_mats is not None):
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(
                list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1
            )
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(self._quats is not None):
            quats = torch.stack(
                list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1
            )
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")
    
    def cuda(self) -> Rotation:
        """
            Analogous to the cuda() method of torch Tensors

            Returns:
                A copy of the Rotation in CUDA memory
        """
        if(self._rot_mats is not None):
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif(self._quats is not None):
            return Rotation(
                rot_mats=None, 
                quats=self._quats.cuda(),
                normalize_quats=False
            )
        else:
            raise ValueError("Both rotations are None")

    def to(self, 
        device: Optional[torch.device], 
        dtype: Optional[torch.dtype]
    ) -> Rotation:
        """
            Analogous to the to() method of torch Tensors

            Args:
                device:
                    A torch device
                dtype:
                    A torch dtype
            Returns:
                A copy of the Rotation using the new device and dtype
        """
        if(self._rot_mats is not None):
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype), 
                quats=None,
            )
        elif(self._quats is not None):
            return Rotation(
                rot_mats=None, 
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> Rotation:
        """
            Returns a copy of the Rotation whose underlying Tensor has been
            detached from its torch graph.

            Returns:
                A copy of the Rotation whose underlying Tensor has been detached
                from its torch graph
        """
        if(self._rot_mats is not None):
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif(self._quats is not None):
            return Rotation(
                rot_mats=None, 
                quats=self._quats.detach(), 
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the 
        shape of the shared batch dimensions of its component parts.
    """
    def __init__(self, 
        rots: Optional[Rotation],
        trans: Optional[torch.Tensor],
    ):
        """
            Args:
                rots: A [*, 3, 3] rotation tensor
                trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if(trans is not None):
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif(rots is not None):
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if(rots is None):
            rots = Rotation.identity(
                batch_dims, dtype, device, requires_grad,
            )
        elif(trans is None):
            trans = identity_trans(
                batch_dims, dtype, device, requires_grad,
            )

        if((rots.shape != trans.shape[:-1]) or
           (rots.device != trans.device)):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.to(dtype=torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int], 
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None, 
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rigid:
        """
            Constructs an identity transformation.

            Args:
                shape: 
                    The desired shape
                dtype: 
                    The dtype of both internal tensors
                device: 
                    The device of both internal tensors
                requires_grad: 
                    Whether grad should be enabled for the internal tensors
            Returns:
                The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, 
        index: Any,
    ) -> Rigid:
        """ 
            Indexes the affine transformation with PyTorch-style indices.
            The index is applied to the shared dimensions of both the rotation
            and the translation.

            E.g.::

                r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
                t = Rigid(r, torch.rand(10, 10, 3))
                indexed = t[3, 4:6]
                assert(indexed.shape == (2,))
                assert(indexed.get_rots().shape == (2,))
                assert(indexed.get_trans().shape == (2, 3))

            Args:
                index: A standard torch tensor index. E.g. 8, (10, None, 3),
                or (3, slice(0, 1, None))
            Returns:
                The indexed tensor 
        """
        if type(index) != tuple:
            index = (index,)
        
        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(self,
        right: torch.Tensor,
    ) -> Rigid:
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rigid.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self,
        left: torch.Tensor,
    ) -> Rigid:
        """
            Reverse pointwise multiplication of the transformation with a 
            tensor.

            Args:
                left:
                    The left multiplicand
            Returns:
                The product
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
            Returns the shape of the shared dimensions of the rotation and
            the translation.
            
            Returns:
                The shape of the transformation
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        """
            Returns the device on which the Rigid's tensors are located.

            Returns:
                The device on which the Rigid's tensors are located
        """
        return self._trans.device

    def get_rots(self) -> Rotation:
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """
            Getter for the translation.

            Returns:
                The stored translation
        """
        return self._trans

    def compose_q_update_vec(self, 
        q_update_vec: torch.Tensor,
    ) -> Rigid:
        """
            Composes the transformation with a quaternion update vector of
            shape [*, 6], where the final 6 columns represent the x, y, and
            z values of a quaternion of form (1, x, y, z) followed by a 3D
            translation.

            Args:
                q_vec: The quaternion update vector.
            Returns:
                The composed transformation.
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec)

        trans_update = self._rots.apply(t_vec)
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(self,
        r: Rigid,
    ) -> Rigid:
        """
            Composes the current rigid object with another.

            Args:
                r:
                    Another Rigid object
            Returns:
                The composition of the two transformations
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, 
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
            Applies the transformation to a coordinate tensor.

            Args:
                pts: A [*, 3] coordinate tensor.
            Returns:
                The transformed points.
        """
        rotated = self._rots.apply(pts) 
        return rotated + self._trans

    def invert_apply(self, 
        pts: torch.Tensor
    ) -> torch.Tensor:
        """
            Applies the inverse of the transformation to a coordinate tensor.

            Args:
                pts: A [*, 3] coordinate tensor
            Returns:
                The transformed points.
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts) 

    def invert(self) -> Rigid:
        """
            Inverts the transformation.

            Returns:
                The inverse transformation.
        """
        rot_inv = self._rots.invert() 
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, 
        fn: Callable[torch.Tensor, torch.Tensor]
    ) -> Rigid:
        """
            Apply a Tensor -> Tensor function to underlying translation and
            rotation tensors, mapping over the translation/rotation dimensions
            respectively.

            Args:
                fn:
                    A Tensor -> Tensor function to be mapped over the Rigid
            Returns:
                The transformed Rigid object
        """     
        new_rots = self._rots.map_tensor_fn(fn) 
        new_trans = torch.stack(
            list(map(fn, torch.unbind(self._trans, dim=-1))), 
            dim=-1
        )

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        """
            Converts a transformation to a homogenous transformation tensor.

            Returns:
                A [*, 4, 4] homogenous transformation tensor
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(
        t: torch.Tensor
    ) -> Rigid:
        """
            Constructs a transformation from a homogenous transformation
            tensor.

            Args:
                t: [*, 4, 4] homogenous transformation tensor
            Returns:
                T object with shape [*]
        """
        if(t.shape[-2:] != (4, 4)):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]
        
        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
            Converts a transformation to a tensor with 7 final columns, four 
            for the quaternion followed by three for the translation.

            Returns:
                A [*, 7] tensor representation of the transformation
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(
        t: torch.Tensor,
        normalize_quats: bool = False,
    ) -> Rigid:
        if(t.shape[-1] != 7):
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(
            rot_mats=None, 
            quats=quats, 
            normalize_quats=normalize_quats
        )

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor, 
        origin: torch.Tensor, 
        p_xy_plane: torch.Tensor, 
        eps: float = 1e-8
    ) -> Rigid:
        """
            Implements algorithm 21. Constructs transformations from sets of 3 
            points using the Gram-Schmidt algorithm.

            Args:
                p_neg_x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(self, 
        dim: int,
    ) -> Rigid:
        """
            Analogous to torch.unsqueeze. The dimension is relative to the
            shared dimensions of the rotation/translation.
            
            Args:
                dim: A positive or negative dimension index.
            Returns:
                The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(
        ts: Sequence[Rigid], 
        dim: int,
    ) -> Rigid:
        """
            Concatenates transformations along a new dimension.

            Args:
                ts: 
                    A list of T objects
                dim: 
                    The dimension along which the transformations should be 
                    concatenated
            Returns:
                A concatenated transformation object
        """
        rots = Rotation.cat([t._rots for t in ts], dim) 
        trans = torch.cat(
            [t._trans for t in ts], dim=dim if dim >= 0 else dim - 1
        )

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable[Rotation, Rotation]) -> Rigid:
        """
            Applies a Rotation -> Rotation function to the stored rotation
            object.

            Args:
                fn: A function of type Rotation -> Rotation
            Returns:
                A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rigid:
        """
            Applies a Tensor -> Tensor function to the stored translation.

            Args:
                fn: 
                    A function of type Tensor -> Tensor to be applied to the
                    translation
            Returns:
                A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """
            Scales the translation by a constant factor.

            Args:
                trans_scale_factor:
                    The constant factor
            Returns:
                A transformation object with a scaled translation.
        """
        fn = lambda t: t * trans_scale_factor
        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self) -> Rigid:
        """
            Detaches the underlying rotation object

            Returns:
                A transformation object with detached rotations
        """
        fn = lambda r: r.detach()
        return self.apply_rot_fn(fn)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """
            Returns a transformation object from reference coordinates.
  
            Note that this method does not take care of symmetries. If you 
            provide the atom positions in the non-standard way, the N atom will 
            end up not at [-0.527250, 1.359329, 0.0] but instead at 
            [-0.527250, -1.359329, 0.0]. You need to take care of such cases in 
            your code.
  
            Args:
                n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
                ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
                c_xyz: A [*, 3] tensor of carbon xyz coordinates.
            Returns:
                A transformation object. After applying the translation and 
                rotation to the reference backbone, the coordinates will 
                approximately equal to the input coordinates.
        """    
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
            Moves the transformation object to GPU memory
            
            Returns:
                A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())


def _build_residue_constants():
    # Copyright 2021 AlQuraishi Laboratory
    # Copyright 2021 DeepMind Technologies Limited
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    """Constants used in AlphaFold."""



    # Internal import (35fd).


    # Distance from one CA to next CA [trans configuration: omega = 180].
    ca_ca = 3.80209737096

    # Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
    # this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
    # chi angles so their chi angle lists are empty.
    chi_angles_atoms = {
        "ALA": [],
        # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
        "ARG": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "NE"],
            ["CG", "CD", "NE", "CZ"],
        ],
        "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
        "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
        "CYS": [["N", "CA", "CB", "SG"]],
        "GLN": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
        ],
        "GLU": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
        ],
        "GLY": [],
        "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
        "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
        "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "LYS": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "CE"],
            ["CG", "CD", "CE", "NZ"],
        ],
        "MET": [
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "SD"],
            ["CB", "CG", "SD", "CE"],
        ],
        "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
        "SER": [["N", "CA", "CB", "OG"]],
        "THR": [["N", "CA", "CB", "OG1"]],
        "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
        "VAL": [["N", "CA", "CB", "CG1"]],
    }

    # If chi angles given in fixed-length array, this matrix determines how to mask
    # them for each AA type. The order is as per restype_order (see below).
    chi_angles_mask = [
        [0.0, 0.0, 0.0, 0.0],  # ALA
        [1.0, 1.0, 1.0, 1.0],  # ARG
        [1.0, 1.0, 0.0, 0.0],  # ASN
        [1.0, 1.0, 0.0, 0.0],  # ASP
        [1.0, 0.0, 0.0, 0.0],  # CYS
        [1.0, 1.0, 1.0, 0.0],  # GLN
        [1.0, 1.0, 1.0, 0.0],  # GLU
        [0.0, 0.0, 0.0, 0.0],  # GLY
        [1.0, 1.0, 0.0, 0.0],  # HIS
        [1.0, 1.0, 0.0, 0.0],  # ILE
        [1.0, 1.0, 0.0, 0.0],  # LEU
        [1.0, 1.0, 1.0, 1.0],  # LYS
        [1.0, 1.0, 1.0, 0.0],  # MET
        [1.0, 1.0, 0.0, 0.0],  # PHE
        [1.0, 1.0, 0.0, 0.0],  # PRO
        [1.0, 0.0, 0.0, 0.0],  # SER
        [1.0, 0.0, 0.0, 0.0],  # THR
        [1.0, 1.0, 0.0, 0.0],  # TRP
        [1.0, 1.0, 0.0, 0.0],  # TYR
        [1.0, 0.0, 0.0, 0.0],  # VAL
    ]

    # The following chi angles are pi periodic: they can be rotated by a multiple
    # of pi without affecting the structure.
    chi_pi_periodic = [
        [0.0, 0.0, 0.0, 0.0],  # ALA
        [0.0, 0.0, 0.0, 0.0],  # ARG
        [0.0, 0.0, 0.0, 0.0],  # ASN
        [0.0, 1.0, 0.0, 0.0],  # ASP
        [0.0, 0.0, 0.0, 0.0],  # CYS
        [0.0, 0.0, 0.0, 0.0],  # GLN
        [0.0, 0.0, 1.0, 0.0],  # GLU
        [0.0, 0.0, 0.0, 0.0],  # GLY
        [0.0, 0.0, 0.0, 0.0],  # HIS
        [0.0, 0.0, 0.0, 0.0],  # ILE
        [0.0, 0.0, 0.0, 0.0],  # LEU
        [0.0, 0.0, 0.0, 0.0],  # LYS
        [0.0, 0.0, 0.0, 0.0],  # MET
        [0.0, 1.0, 0.0, 0.0],  # PHE
        [0.0, 0.0, 0.0, 0.0],  # PRO
        [0.0, 0.0, 0.0, 0.0],  # SER
        [0.0, 0.0, 0.0, 0.0],  # THR
        [0.0, 0.0, 0.0, 0.0],  # TRP
        [0.0, 1.0, 0.0, 0.0],  # TYR
        [0.0, 0.0, 0.0, 0.0],  # VAL
        [0.0, 0.0, 0.0, 0.0],  # UNK
    ]

    # Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
    # psi and chi angles:
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    # The atom positions are relative to the axis-end-atom of the corresponding
    # rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
    # is defined such that the dihedral-angle-definiting atom (the last entry in
    # chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
    # format: [atomname, group_idx, rel_position]
    rigid_group_atom_positions = {
        "ALA": [
            ["N", 0, (-0.525, 1.363, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, -0.000, -0.000)],
            ["CB", 0, (-0.529, -0.774, -1.205)],
            ["O", 3, (0.627, 1.062, 0.000)],
        ],
        "ARG": [
            ["N", 0, (-0.524, 1.362, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, -0.000, -0.000)],
            ["CB", 0, (-0.524, -0.778, -1.209)],
            ["O", 3, (0.626, 1.062, 0.000)],
            ["CG", 4, (0.616, 1.390, -0.000)],
            ["CD", 5, (0.564, 1.414, 0.000)],
            ["NE", 6, (0.539, 1.357, -0.000)],
            ["NH1", 7, (0.206, 2.301, 0.000)],
            ["NH2", 7, (2.078, 0.978, -0.000)],
            ["CZ", 7, (0.758, 1.093, -0.000)],
        ],
        "ASN": [
            ["N", 0, (-0.536, 1.357, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, -0.000, -0.000)],
            ["CB", 0, (-0.531, -0.787, -1.200)],
            ["O", 3, (0.625, 1.062, 0.000)],
            ["CG", 4, (0.584, 1.399, 0.000)],
            ["ND2", 5, (0.593, -1.188, 0.001)],
            ["OD1", 5, (0.633, 1.059, 0.000)],
        ],
        "ASP": [
            ["N", 0, (-0.525, 1.362, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.527, 0.000, -0.000)],
            ["CB", 0, (-0.526, -0.778, -1.208)],
            ["O", 3, (0.626, 1.062, -0.000)],
            ["CG", 4, (0.593, 1.398, -0.000)],
            ["OD1", 5, (0.610, 1.091, 0.000)],
            ["OD2", 5, (0.592, -1.101, -0.003)],
        ],
        "CYS": [
            ["N", 0, (-0.522, 1.362, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.524, 0.000, 0.000)],
            ["CB", 0, (-0.519, -0.773, -1.212)],
            ["O", 3, (0.625, 1.062, -0.000)],
            ["SG", 4, (0.728, 1.653, 0.000)],
        ],
        "GLN": [
            ["N", 0, (-0.526, 1.361, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, 0.000, 0.000)],
            ["CB", 0, (-0.525, -0.779, -1.207)],
            ["O", 3, (0.626, 1.062, -0.000)],
            ["CG", 4, (0.615, 1.393, 0.000)],
            ["CD", 5, (0.587, 1.399, -0.000)],
            ["NE2", 6, (0.593, -1.189, -0.001)],
            ["OE1", 6, (0.634, 1.060, 0.000)],
        ],
        "GLU": [
            ["N", 0, (-0.528, 1.361, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, -0.000, -0.000)],
            ["CB", 0, (-0.526, -0.781, -1.207)],
            ["O", 3, (0.626, 1.062, 0.000)],
            ["CG", 4, (0.615, 1.392, 0.000)],
            ["CD", 5, (0.600, 1.397, 0.000)],
            ["OE1", 6, (0.607, 1.095, -0.000)],
            ["OE2", 6, (0.589, -1.104, -0.001)],
        ],
        "GLY": [
            ["N", 0, (-0.572, 1.337, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.517, -0.000, -0.000)],
            ["O", 3, (0.626, 1.062, -0.000)],
        ],
        "HIS": [
            ["N", 0, (-0.527, 1.360, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, 0.000, 0.000)],
            ["CB", 0, (-0.525, -0.778, -1.208)],
            ["O", 3, (0.625, 1.063, 0.000)],
            ["CG", 4, (0.600, 1.370, -0.000)],
            ["CD2", 5, (0.889, -1.021, 0.003)],
            ["ND1", 5, (0.744, 1.160, -0.000)],
            ["CE1", 5, (2.030, 0.851, 0.002)],
            ["NE2", 5, (2.145, -0.466, 0.004)],
        ],
        "ILE": [
            ["N", 0, (-0.493, 1.373, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.527, -0.000, -0.000)],
            ["CB", 0, (-0.536, -0.793, -1.213)],
            ["O", 3, (0.627, 1.062, -0.000)],
            ["CG1", 4, (0.534, 1.437, -0.000)],
            ["CG2", 4, (0.540, -0.785, -1.199)],
            ["CD1", 5, (0.619, 1.391, 0.000)],
        ],
        "LEU": [
            ["N", 0, (-0.520, 1.363, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, -0.000, -0.000)],
            ["CB", 0, (-0.522, -0.773, -1.214)],
            ["O", 3, (0.625, 1.063, -0.000)],
            ["CG", 4, (0.678, 1.371, 0.000)],
            ["CD1", 5, (0.530, 1.430, -0.000)],
            ["CD2", 5, (0.535, -0.774, 1.200)],
        ],
        "LYS": [
            ["N", 0, (-0.526, 1.362, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, 0.000, 0.000)],
            ["CB", 0, (-0.524, -0.778, -1.208)],
            ["O", 3, (0.626, 1.062, -0.000)],
            ["CG", 4, (0.619, 1.390, 0.000)],
            ["CD", 5, (0.559, 1.417, 0.000)],
            ["CE", 6, (0.560, 1.416, 0.000)],
            ["NZ", 7, (0.554, 1.387, 0.000)],
        ],
        "MET": [
            ["N", 0, (-0.521, 1.364, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, 0.000, 0.000)],
            ["CB", 0, (-0.523, -0.776, -1.210)],
            ["O", 3, (0.625, 1.062, -0.000)],
            ["CG", 4, (0.613, 1.391, -0.000)],
            ["SD", 5, (0.703, 1.695, 0.000)],
            ["CE", 6, (0.320, 1.786, -0.000)],
        ],
        "PHE": [
            ["N", 0, (-0.518, 1.363, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.524, 0.000, -0.000)],
            ["CB", 0, (-0.525, -0.776, -1.212)],
            ["O", 3, (0.626, 1.062, -0.000)],
            ["CG", 4, (0.607, 1.377, 0.000)],
            ["CD1", 5, (0.709, 1.195, -0.000)],
            ["CD2", 5, (0.706, -1.196, 0.000)],
            ["CE1", 5, (2.102, 1.198, -0.000)],
            ["CE2", 5, (2.098, -1.201, -0.000)],
            ["CZ", 5, (2.794, -0.003, -0.001)],
        ],
        "PRO": [
            ["N", 0, (-0.566, 1.351, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.527, -0.000, 0.000)],
            ["CB", 0, (-0.546, -0.611, -1.293)],
            ["O", 3, (0.621, 1.066, 0.000)],
            ["CG", 4, (0.382, 1.445, 0.0)],
            # ['CD', 5, (0.427, 1.440, 0.0)],
            ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
        ],
        "SER": [
            ["N", 0, (-0.529, 1.360, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, -0.000, -0.000)],
            ["CB", 0, (-0.518, -0.777, -1.211)],
            ["O", 3, (0.626, 1.062, -0.000)],
            ["OG", 4, (0.503, 1.325, 0.000)],
        ],
        "THR": [
            ["N", 0, (-0.517, 1.364, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.526, 0.000, -0.000)],
            ["CB", 0, (-0.516, -0.793, -1.215)],
            ["O", 3, (0.626, 1.062, 0.000)],
            ["CG2", 4, (0.550, -0.718, -1.228)],
            ["OG1", 4, (0.472, 1.353, 0.000)],
        ],
        "TRP": [
            ["N", 0, (-0.521, 1.363, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.525, -0.000, 0.000)],
            ["CB", 0, (-0.523, -0.776, -1.212)],
            ["O", 3, (0.627, 1.062, 0.000)],
            ["CG", 4, (0.609, 1.370, -0.000)],
            ["CD1", 5, (0.824, 1.091, 0.000)],
            ["CD2", 5, (0.854, -1.148, -0.005)],
            ["CE2", 5, (2.186, -0.678, -0.007)],
            ["CE3", 5, (0.622, -2.530, -0.007)],
            ["NE1", 5, (2.140, 0.690, -0.004)],
            ["CH2", 5, (3.028, -2.890, -0.013)],
            ["CZ2", 5, (3.283, -1.543, -0.011)],
            ["CZ3", 5, (1.715, -3.389, -0.011)],
        ],
        "TYR": [
            ["N", 0, (-0.522, 1.362, 0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.524, -0.000, -0.000)],
            ["CB", 0, (-0.522, -0.776, -1.213)],
            ["O", 3, (0.627, 1.062, -0.000)],
            ["CG", 4, (0.607, 1.382, -0.000)],
            ["CD1", 5, (0.716, 1.195, -0.000)],
            ["CD2", 5, (0.713, -1.194, -0.001)],
            ["CE1", 5, (2.107, 1.200, -0.002)],
            ["CE2", 5, (2.104, -1.201, -0.003)],
            ["OH", 5, (4.168, -0.002, -0.005)],
            ["CZ", 5, (2.791, -0.001, -0.003)],
        ],
        "VAL": [
            ["N", 0, (-0.494, 1.373, -0.000)],
            ["CA", 0, (0.000, 0.000, 0.000)],
            ["C", 0, (1.527, -0.000, -0.000)],
            ["CB", 0, (-0.533, -0.795, -1.213)],
            ["O", 3, (0.627, 1.062, -0.000)],
            ["CG1", 4, (0.540, 1.429, -0.000)],
            ["CG2", 4, (0.533, -0.776, 1.203)],
        ],
    }

    # A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
    residue_atoms = {
        "ALA": ["C", "CA", "CB", "N", "O"],
        "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
        "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
        "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
        "CYS": ["C", "CA", "CB", "N", "O", "SG"],
        "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
        "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
        "GLY": ["C", "CA", "N", "O"],
        "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
        "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
        "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
        "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
        "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
        "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
        "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
        "SER": ["C", "CA", "CB", "N", "O", "OG"],
        "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
        "TRP": [
            "C",
            "CA",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
            "N",
            "NE1",
            "O",
        ],
        "TYR": [
            "C",
            "CA",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "N",
            "O",
            "OH",
        ],
        "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
    }

    # Naming swaps for ambiguous atom names.
    # Due to symmetries in the amino acids the naming of atoms is ambiguous in
    # 4 of the 20 amino acids.
    # (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
    # in LEU, VAL and ARG can be resolved by using the 3d constellations of
    # the 'ambiguous' atoms and their neighbours)
    # TODO: ^ interpret this
    residue_atom_renaming_swaps = {
        "ASP": {"OD1": "OD2"},
        "GLU": {"OE1": "OE2"},
        "PHE": {"CD1": "CD2", "CE1": "CE2"},
        "TYR": {"CD1": "CD2", "CE1": "CE2"},
    }

    # Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
    van_der_waals_radius = {
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "S": 1.8,
    }

    Bond = collections.namedtuple(
        "Bond", ["atom1_name", "atom2_name", "length", "stddev"]
    )
    BondAngle = collections.namedtuple(
        "BondAngle",
        ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
    )




    # Between-residue bond lengths for general bonds (first element) and for Proline
    # (second element).
    between_res_bond_length_c_n = [1.329, 1.341]
    between_res_bond_length_stddev_c_n = [0.014, 0.016]

    # Between-residue cos_angles.
    between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
    between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

    # This mapping is used when we need to store atom data in a format that requires
    # fixed atom data size for every residue (e.g. a numpy array).
    atom_types = [
        "N",
        "CA",
        "C",
        "CB",
        "O",
        "CG",
        "CG1",
        "CG2",
        "OG",
        "OG1",
        "SG",
        "CD",
        "CD1",
        "CD2",
        "ND1",
        "ND2",
        "OD1",
        "OD2",
        "SD",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "NE",
        "NE1",
        "NE2",
        "OE1",
        "OE2",
        "CH2",
        "NH1",
        "NH2",
        "OH",
        "CZ",
        "CZ2",
        "CZ3",
        "NZ",
        "OXT",
    ]
    atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
    atom_type_num = len(atom_types)  # := 37.

    # A compact atom encoding with 14 columns
    # pylint: disable=line-too-long
    # pylint: disable=bad-whitespace
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "OD1",
            "ND2",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "ASP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "OD1",
            "OD2",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG1",
            "CG2",
            "CD1",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "LEU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "LYS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "CE",
            "NZ",
            "",
            "",
            "",
            "",
            "",
        ],
        "MET": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "SD",
            "CE",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG1",
            "CG2",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG1",
            "CG2",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }
    # pylint: enable=line-too-long
    # pylint: enable=bad-whitespace


    # This is the standard residue order when coding AA type as a number.
    # Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
    restypes = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    restype_order = {restype: i for i, restype in enumerate(restypes)}
    restype_num = len(restypes)  # := 20.
    unk_restype_index = restype_num  # Catch-all index for unknown restypes.

    restypes_with_x = restypes + ["X"]
    restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


    def sequence_to_onehot(
        sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False
    ) -> np.ndarray:
        """Maps the given sequence into a one-hot encoded matrix.

        Args:
          sequence: An amino acid sequence.
          mapping: A dictionary mapping amino acids to integers.
          map_unknown_to_x: If True, any amino acid that is not in the mapping will be
            mapped to the unknown amino acid 'X'. If the mapping doesn't contain
            amino acid 'X', an error will be thrown. If False, any amino acid not in
            the mapping will throw an error.

        Returns:
          A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
          the sequence.

        Raises:
          ValueError: If the mapping doesn't contain values from 0 to
            num_unique_aas - 1 without any gaps.
        """
        num_entries = max(mapping.values()) + 1

        if sorted(set(mapping.values())) != list(range(num_entries)):
            raise ValueError(
                "The mapping must have values from 0 to num_unique_aas-1 "
                "without any gaps. Got: %s" % sorted(mapping.values())
            )

        one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

        for aa_index, aa_type in enumerate(sequence):
            if map_unknown_to_x:
                if aa_type.isalpha() and aa_type.isupper():
                    aa_id = mapping.get(aa_type, mapping["X"])
                else:
                    raise ValueError(
                        f"Invalid character in the sequence: {aa_type}"
                    )
            else:
                aa_id = mapping[aa_type]
            one_hot_arr[aa_index, aa_id] = 1

        return one_hot_arr


    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }


    # NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
    # 1-to-1 mapping of 3 letter names to one letter names. The latter contains
    # many more, and less common, three letter names as keys and maps many of these
    # to the same one letter name (including 'X' and 'U' which we don't use here).
    restype_3to1 = {v: k for k, v in restype_1to3.items()}

    # Define a restype name for all unknown residues.
    unk_restype = "UNK"

    resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
    resname_to_idx = {resname: i for i, resname in enumerate(resnames)}


    # The mapping here uses hhblits convention, so that B is mapped to D, J and O
    # are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
    # remaining 20 amino acids are kept in alphabetical order.
    # There are 2 non-amino acid codes, X (representing any amino acid) and
    # "-" representing a missing amino acid in an alignment.  The id for these
    # codes is put at the end (20 and 21) so that they can easily be ignored if
    # desired.
    HHBLITS_AA_TO_ID = {
        "A": 0,
        "B": 2,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "J": 20,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "O": 20,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "U": 1,
        "V": 17,
        "W": 18,
        "X": 20,
        "Y": 19,
        "Z": 3,
        "-": 21,
    }

    # Partial inversion of HHBLITS_AA_TO_ID.
    ID_TO_HHBLITS_AA = {
        0: "A",
        1: "C",  # Also U.
        2: "D",  # Also B.
        3: "E",  # Also Z.
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",  # Includes J and O.
        21: "-",
    }

    restypes_with_x_and_gap = restypes + ["X", "-"]
    MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
        restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i])
        for i in range(len(restypes_with_x_and_gap))
    )


    def _make_standard_atom_mask() -> np.ndarray:
        """Returns [num_res_types, num_atom_types] mask array."""
        # +1 to account for unknown (all 0s).
        mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
        for restype, restype_letter in enumerate(restypes):
            restype_name = restype_1to3[restype_letter]
            atom_names = residue_atoms[restype_name]
            for atom_name in atom_names:
                atom_type = atom_order[atom_name]
                mask[restype, atom_type] = 1
        return mask


    STANDARD_ATOM_MASK = _make_standard_atom_mask()


    # A one hot representation for the first and second atoms defining the axis
    # of rotation for each chi-angle in each residue.
    def chi_angle_atom(atom_index: int) -> np.ndarray:
        """Define chi-angle rigid groups via one-hot representations."""
        chi_angles_index = {}
        one_hots = []

        for k, v in chi_angles_atoms.items():
            indices = [atom_types.index(s[atom_index]) for s in v]
            indices.extend([-1] * (4 - len(indices)))
            chi_angles_index[k] = indices

        for r in restypes:
            res3 = restype_1to3[r]
            one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
            one_hots.append(one_hot)

        one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
        one_hot = np.stack(one_hots, axis=0)
        one_hot = np.transpose(one_hot, [0, 2, 1])

        return one_hot


    chi_atom_1_one_hot = chi_angle_atom(1)
    chi_atom_2_one_hot = chi_angle_atom(2)

    # An array like chi_angles_atoms but using indices rather than names.
    chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
    # Fixed residue -> chi angle -> atom-name list; no generic tree mapper needed.
    chi_angles_atom_indices = [
        [[atom_order[atom_name] for atom_name in chi] for chi in residue_chis]
        for residue_chis in chi_angles_atom_indices
    ]
    chi_angles_atom_indices = np.array(
        [
            chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
            for chi_atoms in chi_angles_atom_indices
        ]
    )

    # Mapping from (res_name, atom_name) pairs to the atom's chi group index
    # and atom index within that group.
    chi_groups_for_atom = collections.defaultdict(list)
    for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
        for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
            for atom_i, atom in enumerate(chi_group):
                chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
    chi_groups_for_atom = dict(chi_groups_for_atom)


    def _make_rigid_transformation_4x4(ex, ey, translation):
        """Create a rigid 4x4 transformation matrix from two axes and transl."""
        # Normalize ex.
        ex_normalized = ex / np.linalg.norm(ex)

        # make ey perpendicular to ex
        ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
        ey_normalized /= np.linalg.norm(ey_normalized)

        # compute ez as cross product
        eznorm = np.cross(ex_normalized, ey_normalized)
        m = np.stack(
            [ex_normalized, ey_normalized, eznorm, translation]
        ).transpose()
        m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
        return m


    # create an array with (restype, atomtype) --> rigid_group_idx
    # and an array with (restype, atomtype, coord) for the atom positions
    # and compute affine transformation matrices (4,4) from one rigid group to the
    # previous group
    restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
    restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
    restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
    restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
    restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


    def _make_rigid_group_constants():
        """Fill the arrays above."""
        for restype, restype_letter in enumerate(restypes):
            resname = restype_1to3[restype_letter]
            for atomname, group_idx, atom_position in rigid_group_atom_positions[
                resname
            ]:
                atomtype = atom_order[atomname]
                restype_atom37_to_rigid_group[restype, atomtype] = group_idx
                restype_atom37_mask[restype, atomtype] = 1
                restype_atom37_rigid_group_positions[
                    restype, atomtype, :
                ] = atom_position

                atom14idx = restype_name_to_atom14_names[resname].index(atomname)
                restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
                restype_atom14_mask[restype, atom14idx] = 1
                restype_atom14_rigid_group_positions[
                    restype, atom14idx, :
                ] = atom_position

        for restype, restype_letter in enumerate(restypes):
            resname = restype_1to3[restype_letter]
            atom_positions = {
                name: np.array(pos)
                for name, _, pos in rigid_group_atom_positions[resname]
            }

            # backbone to backbone is the identity transform
            restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

            # pre-omega-frame to backbone (currently dummy identity matrix)
            restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

            # phi-frame to backbone
            mat = _make_rigid_transformation_4x4(
                ex=atom_positions["N"] - atom_positions["CA"],
                ey=np.array([1.0, 0.0, 0.0]),
                translation=atom_positions["N"],
            )
            restype_rigid_group_default_frame[restype, 2, :, :] = mat

            # psi-frame to backbone
            mat = _make_rigid_transformation_4x4(
                ex=atom_positions["C"] - atom_positions["CA"],
                ey=atom_positions["CA"] - atom_positions["N"],
                translation=atom_positions["C"],
            )
            restype_rigid_group_default_frame[restype, 3, :, :] = mat

            # chi1-frame to backbone
            if chi_angles_mask[restype][0]:
                base_atom_names = chi_angles_atoms[resname][0]
                base_atom_positions = [
                    atom_positions[name] for name in base_atom_names
                ]
                mat = _make_rigid_transformation_4x4(
                    ex=base_atom_positions[2] - base_atom_positions[1],
                    ey=base_atom_positions[0] - base_atom_positions[1],
                    translation=base_atom_positions[2],
                )
                restype_rigid_group_default_frame[restype, 4, :, :] = mat

            # chi2-frame to chi1-frame
            # chi3-frame to chi2-frame
            # chi4-frame to chi3-frame
            # luckily all rotation axes for the next frame start at (0,0,0) of the
            # previous frame
            for chi_idx in range(1, 4):
                if chi_angles_mask[restype][chi_idx]:
                    axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                    axis_end_atom_position = atom_positions[axis_end_atom_name]
                    mat = _make_rigid_transformation_4x4(
                        ex=axis_end_atom_position,
                        ey=np.array([-1.0, 0.0, 0.0]),
                        translation=axis_end_atom_position,
                    )
                    restype_rigid_group_default_frame[
                        restype, 4 + chi_idx, :, :
                    ] = mat


    _make_rigid_group_constants()




    restype_atom14_ambiguous_atoms = np.zeros((21, 14), dtype=np.float32)
    restype_atom14_ambiguous_atoms_swap_idx = np.tile(
        np.arange(14, dtype=int), (21, 1)
    )


    def _make_atom14_ambiguity_feats():
        for res, pairs in residue_atom_renaming_swaps.items():
            res_idx = restype_order[restype_3to1[res]]
            for atom1, atom2 in pairs.items():
                atom1_idx = restype_name_to_atom14_names[res].index(atom1)
                atom2_idx = restype_name_to_atom14_names[res].index(atom2)
                restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
                restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
                restype_atom14_ambiguous_atoms_swap_idx[
                    res_idx, atom1_idx
                ] = atom2_idx
                restype_atom14_ambiguous_atoms_swap_idx[
                    res_idx, atom2_idx
                ] = atom1_idx


    _make_atom14_ambiguity_feats()


    def aatype_to_str_sequence(aatype):
        return ''.join([
            restypes_with_x[aatype[i]] 
            for i in range(len(aatype))
        ])

    return SimpleNamespace(**locals())

rc = _build_residue_constants()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # rc.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB 
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if(remark is not None):
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if(parents_chain_index is not None):
        parents = [
            p for i, p in zip(parents_chain_index, parents) if i == chain_id
        ]

    if(parents is None or len(parents) == 0):
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = rc.restypes + ["X"]
    res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], "UNK")
    atom_types = rc.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    if np.any(aatype > rc.restype_num):
        raise ValueError("Invalid aatypes.")

    headers = get_pdb_headers(prot)
    if(len(headers) > 0):
        pdb_lines.extend(headers)

    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase
    # Add all atom sites.
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
    
            chain_tag = "A"
            if(chain_index is not None):
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = (i == n - 1)
        if(chain_index is not None):
            if(i != n - 1 and chain_index[i + 1] != prev_chain_index):
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if(should_terminate):
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if(i != n - 1):
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """
    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return Protein(
        aatype=features["aatype"],
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # alpha = alpha.unsqueeze(1).repeat(1,default_r.shape[1],1,1,1)
    if len(default_r.shape) == 4:   #[batch,k,L,8]  
        alpha = alpha.unsqueeze(1).repeat(1,default_r.shape[1],1,1,1)
    # different indices.
    # different indices.
    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


def hu_model_pred_to_atom14_pos(quaternion,translation,angles,aatype):
    of_rigid_q = torch.cat((quaternion[...,0].unsqueeze(-1),-quaternion[...,1:]),dim=-1)    # 胡俭师兄和openfold所规定的q有差异，转为openfold规定下的q
    of_bb_rigid_tensor_7 = torch.cat((of_rigid_q,translation),dim=-1)
    of_bb_rigid = Rigid.from_tensor_7(of_bb_rigid_tensor_7)
    k_angles = angles  #[batch,k,L,7,2]
    k_seq = aatype          #[batch,k,L]
    rrgdf=torch.tensor(
                    rc.restype_rigid_group_default_frame,
                    dtype=torch.float,
                    device=quaternion.device,
                    requires_grad=False,
                )
    group_idx=torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=quaternion.device,
                    requires_grad=False,
                )
    atom_mask=torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=torch.long,
                    device=quaternion.device,
                    requires_grad=False,
                )
    lit_positions=torch.tensor(
            rc.restype_atom14_rigid_group_positions,
            dtype=torch.float,
            device=quaternion.device,
            requires_grad=False,
            )
    of_all_frames_to_global = torsion_angles_to_frames(
            of_bb_rigid,
            k_angles,
            k_seq,
            rrgdf

        )
    of_pred_all_atoms = frames_and_literature_positions_to_atom14_pos(
            of_all_frames_to_global,
            k_seq,
            rrgdf,group_idx,atom_mask,lit_positions
        )
    return of_pred_all_atoms


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein['aatype'].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


"""Reusable helpers for Cerebra_Seq inference scripts."""





HHBLITS_AA_TO_ID = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    "-": 21,
}

_ANCHOR_CLUSTERS_A100 = {
    (0, 96): 18,
    (96, 224): 24,
    (224, 324): 32,
    (324, 588): 48,
    (588, 700): 48,
    (700, 900): 48,
    (900, 1000): 48,
}

def cerebra_autocast(device, use_bf16):
    device = torch.device(device)
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if use_bf16 and device.type == "cuda" and bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def clear_cuda_cache(device=None, synchronize=False):
    if not torch.cuda.is_available():
        return
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            return
        index = device.index if device.index is not None else torch.cuda.current_device()
        if synchronize:
            torch.cuda.synchronize(index)
    torch.cuda.empty_cache()


def ensure_model_on_device(model, device):
    device = torch.device(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()


def offload_model_to_cpu(model, previous_device, clear_cache=True):
    if next(model.parameters()).device.type != "cpu":
        model.to("cpu")
    model.eval()
    if clear_cache:
        clear_cuda_cache(previous_device, synchronize=True)


def offload_esm_models(
    esm3_model,
    esmc_model,
    device_esm3,
    device_esmc,
    clear_cache=True,
):
    offload_model_to_cpu(esm3_model, device_esm3, clear_cache=clear_cache)
    offload_model_to_cpu(esmc_model, device_esmc, clear_cache=clear_cache)


def should_offload_esm(seq_len, offload_length):
    return offload_length is not None and offload_length >= 0 and seq_len > offload_length


def read_fasta_sequence(path):
    """Read one protein sequence and normalize its case."""
    sequence = "".join(line.strip() for line in Path(path).read_text().splitlines() if not line.startswith(">"))
    if not sequence:
        raise ValueError(f"Empty FASTA: {path}")
    return sequence.upper()


def load_cerebra_model(device, checkpoint="model1", revision=None, cache_dir=None, training=False):
    """Load HF weights, optionally configuring gradient checkpointing for training.

    The caller controls train/eval mode, gradients and checkpoint restoration.
    """
    from transformers import AutoConfig, AutoModel

    kwargs = dict(revision=revision, cache_dir=cache_dir, trust_remote_code=True)
    if training:
        config = AutoConfig.from_pretrained("Gonglab/Cerebra_Seq", **kwargs)
        config.core_config["globals"].update(blocks_per_ckpt=1, chunk_size=None, use_lma=False, offload_inference=False)
        config.core_config["model"]["evoformer_stack"]["blocks_per_ckpt"] = 1
        kwargs["config"] = config
    return AutoModel.from_pretrained("Gonglab/Cerebra_Seq", checkpoint=checkpoint, device=str(device), **kwargs).float()


def build_cerebra_batch(sequence, esmc, esm3, device, batched=False):
    """Assemble residue inputs without detaching the embedding tensors."""
    length = len(sequence)
    batch = {
        "X1D_esm_c": esmc.to(device=device, dtype=torch.float32),
        "X1D_esm3": esm3.to(device=device, dtype=torch.float32),
        "target_feat": sequence_to_hhblits_ids(sequence, device),
        "residue_index": torch.arange(1, length + 1, dtype=torch.long, device=device),
    }
    return {key: value.unsqueeze(0) for key, value in batch.items()} if batched else batch


def prepare_cerebra_inputs(batch, model, clone=False):
    """Prepare batched model inputs while preserving integer IDs and gradients."""
    keys = ("target_feat", "residue_index", "X1D_esm3", "X1D_esm_c")
    feats = {key: batch[key].clone() if clone else batch[key] for key in keys}
    feats = move_batch_to_model(feats, model)
    feats["seq_mask"] = torch.ones(feats["X1D_esm_c"].shape[:2], device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    return feats


def sequence_to_hhblits_ids(sequence, device):
    return torch.tensor(
        [HHBLITS_AA_TO_ID.get(residue.upper(), 20) for residue in sequence],
        dtype=torch.long,
        device=device,
    )


def select_anchor_indices(length):
    """Select anchors by length using the original A100 default configuration."""
    n_clusters = 24
    for (min_length, max_length), value in _ANCHOR_CLUSTERS_A100.items():
        if min_length < length <= max_length:
            n_clusters = value
            break

    anchors = np.array(
        [int(index * (length - 8) / n_clusters) for index in range(n_clusters)]
    ) + 5
    anchors = np.clip(anchors, a_min=2, a_max=length - 2).astype(int)
    if n_clusters + 8 > length:
        anchors = np.arange(2, length - 2).astype(int)
    return anchors


def move_batch_to_model(batch, model):
    parameter = next(model.parameters())
    converted = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            value = value.to(
                device=parameter.device,
                dtype=parameter.dtype,
                non_blocking=True,
            )
        else:
            value = value.to(device=parameter.device, non_blocking=True)
        converted[key] = value
    return converted


def NormQuaternion(quaternion):
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    eps = torch.finfo(quaternion.dtype).eps
    quaternion = quaternion / norm.clamp_min(eps)
    sign = torch.where(quaternion[..., :1] < 0, -1, 1)
    return sign * quaternion


def QuaternionMM(left, right):
    scalar = left[..., 0] * right[..., 0] - (
        left[..., 1:] * right[..., 1:]
    ).sum(dim=-1)
    vector = (
        torch.cross(right[..., 1:], left[..., 1:], dim=-1)
        + left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
    )
    return torch.cat((scalar.unsqueeze(-1), vector), dim=-1)



def NormQuaternionMM(q1, q2):
    return NormQuaternion(QuaternionMM(q1, q2))


def Rotation2Quaternion(r):
    """Convert rotation matrices using utils while preserving the original API."""
    return NormQuaternion(_rotation_matrix_to_quaternion(r))


def NormVec(V):
    eps = 1e-7
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= (torch.norm(axis_x, dim=-1).unsqueeze(1) + eps)
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= (torch.norm(axis_z, dim=-1).unsqueeze(1) + eps)
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= (torch.norm(axis_y, dim=-1).unsqueeze(1) + eps)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec


def PsiPhi(atoms):
    eps = 1e-7
    def psi(CA, C, N):
        a = N[1:] - C[:-1]
        b = C - CA
        c = N - CA
        ab = torch.linalg.cross(a, b[:-1])
        bc = torch.linalg.cross(b[:-1], c[:-1])
        ca = torch.linalg.cross(c[:-1], a)
        
        cos_ca_b = torch.sum(ca * b[:-1], dim=-1) / (torch.linalg.norm(ca, dim=-1) * torch.linalg.norm(b[:-1], dim=-1) + eps)
        cospsi = torch.sum(ab * bc, dim=-1)/(torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) + eps)
        cospsi = np.pi - torch.arccos(torch.clamp(cospsi, max=1, min=-1))
        return (cos_ca_b / abs(cos_ca_b)) * cospsi

    def phi(CA, C, N):
        b = C - CA
        c = N - CA
        d = C[:-1] - N[1:]
        bc = torch.linalg.cross(b[1:], c[1:])
        cd = torch.linalg.cross(c[1:], d)
        bd = torch.linalg.cross(b[1:], d)
        cos_bd_c = torch.sum(bd * c[1:], dim=-1) / (torch.linalg.norm(bd, dim=-1) * torch.linalg.norm(c[1:], dim=-1) + eps)
        cosphi = torch.sum(bc * cd, dim=-1) / (torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(cd, dim=-1) + eps)
        cosphi = np.pi - torch.arccos(torch.clamp(cosphi, max=1, min=-1))
        return (cos_bd_c / abs(cos_bd_c)) * cosphi
    N, CA, C = atoms[:, 2], atoms[:, 0], atoms[:, 1]
    return torch.stack([psi(CA, C, N), phi(CA, C, N)], dim=1)


def comp_label(atoms):
    eps = 1e-7
    nres = atoms.shape[0]
    N_CA_C = atoms[:, [2, 0, 1], :].reshape(-1, 3, 3)
    rotation = NormVec(N_CA_C)
    U, _, V = torch.svd(torch.eye(3).unsqueeze(0).permute(0, 2, 1) @ rotation)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)))
    Id = torch.eye(3).repeat(nres, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))
    q = Rotation2Quaternion(r)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    QAll = NormQuaternionMM(q.unsqueeze(1).repeat(1, nres, 1), q_1.unsqueeze(0).repeat(nres, 1, 1))
    
    QAll[..., 0][torch.isnan(QAll[..., 0])] = 1.
    QAll[torch.isnan(QAll)] = 0.
    QAll = NormQuaternion(QAll)
    
    xyz_CA = torch.einsum('a b i, a i j -> a b j', atoms[:, 0].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_C  = torch.einsum('a b i, a i j -> a b j', atoms[:, 1].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_N  = torch.einsum('a b i, a i j -> a b j', atoms[:, 2].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_CB = torch.einsum('a b i, a i j -> a b j', atoms[:, 3].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    
    CA_C_N_CB = torch.stack([xyz_CA, xyz_C, xyz_N, xyz_CB], dim=-2)
    
    CA = atoms[:, 0].unsqueeze(0) - atoms[:, 0].unsqueeze(1)
    r_CA = torch.sqrt((CA * CA).sum(-1) + eps)
    CB_dist = atoms[:, 3, :].unsqueeze(0) - atoms[:, 3, :].unsqueeze(1)
    CB_dist = torch.sqrt((CB_dist * CB_dist).sum(-1))
    CB_dist = (CB_dist*2 - 7).long()
    CB_dist = CB_dist.clamp(min=0, max=35)
    # CB_dist = (torch.stack(CB_dist) <= 8).long().view(-1)
    psi_phi = PsiPhi(atoms)

    return CA_C_N_CB, CB_dist, r_CA, QAll, psi_phi


def _kabsch_align(mobile, target):
    mobile = mobile - mobile.mean(axis=0)
    target = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(np.dot(mobile.T, target))
    handedness = np.sign(np.linalg.det(np.dot(left, right)))
    correction = np.diag((1.0, 1.0, handedness))
    rotation = np.dot(right.T, np.dot(correction, left.T))
    return np.dot(target, rotation), rotation


def _mean_closest_positions(positions, top_k=3):
    mean_position = positions.mean(dim=0)
    distances = torch.abs(positions - mean_position).mean(dim=-1)
    k = min(top_k, positions.shape[0])
    threshold = distances.topk(k=k, largest=False).values[-1]
    return positions[distances <= threshold].mean(dim=0)


def tensor_to_numpy(tensor):
    """Move a tensor to CPU in a NumPy-compatible dtype."""
    tensor = tensor.detach()
    if torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor.cpu().numpy()


def AnchorFrameConsensus(outputs, main_anchor_id, top_k=3):
    translations = outputs["translation"][-1]
    quaternions = outputs["quaternion"][-1]
    consensus_dtype = (
        torch.float32
        if quaternions.dtype in (torch.float16, torch.bfloat16)
        else quaternions.dtype
    )
    xyz = tensor_to_numpy(translations[0])
    main_anchor_position = xyz[main_anchor_id]

    aligned_positions = []
    frame_rotations = []
    for anchor_position in xyz:
        aligned, rotation = _kabsch_align(main_anchor_position, anchor_position)
        aligned_positions.append(aligned)
        frame_rotations.append(rotation)

    aligned_positions = torch.as_tensor(
        np.stack(aligned_positions), device=quaternions.device, dtype=consensus_dtype
    ).permute(1, 0, 2)
    consensus_translation = torch.stack(
        [_mean_closest_positions(position, top_k) for position in aligned_positions]
    )

    frame_rotations = torch.as_tensor(
        np.stack(frame_rotations), device=quaternions.device, dtype=consensus_dtype
    )
    frame_quaternions = Rotation2Quaternion(frame_rotations)[
        None, :, None, :
    ]
    combined_quaternions = QuaternionMM(
        frame_quaternions, quaternions.to(dtype=consensus_dtype)
    )
    combined_quaternions = combined_quaternions[0].permute(1, 0, 2)
    consensus_quaternion = torch.stack(
        [_mean_closest_positions(quaternion, top_k) for quaternion in combined_quaternions]
    )

    return (
        NormQuaternion(consensus_quaternion.unsqueeze(0)),
        consensus_translation.unsqueeze(0),
    )


def reduce_plddt_output(raw_plddt):
    if raw_plddt.dim() == 2:
        return raw_plddt.float()
    if raw_plddt.dim() == 3:
        return raw_plddt.mean(dim=1).float()

    num_bins = raw_plddt.shape[-1]
    bounds = (torch.arange(num_bins, device=raw_plddt.device, dtype=raw_plddt.dtype) + 0.5) / num_bins
    probabilities = torch.softmax(raw_plddt, dim=-1)
    plddt = torch.sum(probabilities * bounds, dim=-1)
    return plddt.mean(dim=-1).mean(dim=1).float()


def parse_fasta_file(fasta_path):
    entries = []
    current_name = None
    current_sequence = []
    path = Path(fasta_path)

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                entries.append((current_name, "".join(current_sequence)))
            description = line[1:].strip()
            current_name = description.split()[0] if description else path.stem
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_name is not None:
        entries.append((current_name, "".join(current_sequence)))
    elif current_sequence:
        entries.append((path.stem, "".join(current_sequence)))
    return entries


def collect_fasta_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted((*path.glob("*.fa"), *path.glob("*.fasta")))
    raise FileNotFoundError(f"Input {path} not found")


def write_feature_pt(path, features):
    """Save a CPU tensor dictionary, readable with torch.load(weights_only=True)."""
    dtypes = {
        "relaxed": np.bool_,
        "esmc": np.float32,
        "esm3": np.float32,
        "node_embedding": np.float32,
        "node_embedding_HC": np.float32,
        "edge_embedding": np.float32,
        "plddt": np.float16,
        "aatype": np.int8,
        "residue_index": np.int32,
        "final_atom_positions": np.float32,
        "final_atom_mask": np.int8,
    }
    optional = {"esmc", "esm3", "node_embedding_HC", "relaxed"}
    missing = set(dtypes) - optional - set(features)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    tensors = {}
    for name, dtype in dtypes.items():
        if name in optional and name not in features:
            continue
        value = features[name]
        if torch.is_tensor(value):
            value = tensor_to_numpy(value)
        tensors[name] = torch.from_numpy(np.asarray(value, dtype=dtype).copy())
    # Publish only complete files so interrupted writes are not skipped on rerun.
    from tempfile import NamedTemporaryFile

    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(tensors, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=None)
def get_relax_backend(device="cuda:0"):
    """Check optional Amber dependencies and a working OpenMM platform lazily."""
    import importlib

    errors = []
    for package in ("Cerebra_Seq", "CerebraSeq", "openfold"):
        try:
            module = importlib.import_module(f"{package}.np.relax.relax")
            protein_module = importlib.import_module(f"{package}.np.protein")
            openmm = module.amber_minimize.openmm
            # Confirm the force field resources, not just the Python imports.
            module.amber_minimize.openmm_app.ForceField("amber99sb.xml")
        except Exception as error:
            errors.append(f"{package}: {error}")
            continue
        device = torch.device(device)
        platforms = ["CUDA", "CPU"] if device.type == "cuda" else ["CPU"]
        for name in platforms:
            try:
                platform = openmm.Platform.getPlatformByName(name)
                properties = {}
                if name == "CUDA":
                    properties["DeviceIndex"] = str(device.index if device.index is not None else 0)
                system = openmm.System()
                system.addParticle(1.0)
                integrator = openmm.VerletIntegrator(0.001)
                context = openmm.Context(system, integrator, platform, properties)
                del context, integrator
                return module.AmberRelaxation, protein_module, platform, properties
            except Exception as error:
                errors.append(f"{package}/{name}: {error}")
    raise RuntimeError(
        "--relax requested, but the current Python environment has no usable Amber relaxation backend. "
        "It requires Cerebra_Seq/CerebraSeq or OpenFold with its relaxation dependencies "
        "(including OpenMM and pdbfixer). Details: " + "; ".join(errors)
    )


def check_relax_environment(device):
    _, _, platform, _ = get_relax_backend(str(device))
    print(f"[relax] Available: OpenMM {platform.getName()}", flush=True)


def relax_prediction(protein, device):
    """Use the original Cerebra Amber settings and return an atom37 protein."""
    relaxer_class, protein_module, platform, properties = get_relax_backend(str(device))
    relaxer = relaxer_class(
        max_iterations=0, tolerance=2.39, stiffness=10.0,
        exclude_residues=[], max_outer_iterations=20,
        use_gpu=platform.getName() == "CUDA",
    )
    # The original backend creates its own Context with default properties.
    # Select the requested visible GPU without changing CUDA_VISIBLE_DEVICES.
    previous = {key: platform.getPropertyDefaultValue(key) for key in properties}
    try:
        for key, value in properties.items():
            platform.setPropertyDefaultValue(key, value)
        pdb_string, _, _ = relaxer.process(prot=protein)
    finally:
        for key, value in previous.items():
            platform.setPropertyDefaultValue(key, value)
    relaxed = protein_module.from_pdb_string(pdb_string)
    if not np.array_equal(relaxed.aatype, protein.aatype) or not np.array_equal(
        relaxed.residue_index, protein.residue_index
    ):
        raise ValueError("Relaxation changed residue identity or ordering")
    return relaxed


def feature_output_complete(path, require_relax=False):
    path = Path(path)
    if not path.is_file():
        return False
    if not require_relax:
        return True
    try:
        features = torch.load(path, map_location="cpu", weights_only=True)
        return bool(features.get("relaxed", False))
    except Exception:
        return False
