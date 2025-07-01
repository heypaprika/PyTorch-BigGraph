# ✅ 모든 CUDA 오류 방지를 위한 torch.device 통일 수정 버전

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from torchbiggraph.plugin import PluginRegistry
from torchbiggraph.types import FloatTensorType, LongTensorType, Side
from torchbiggraph.util import match_shape


class AbstractOperator(nn.Module, ABC):
    """Perform the same operation on many vectors.

    Given a tensor containing a set of vectors, perform the same operation on
    all of them, with a common set of parameters. The dimension of these vectors
    will be given at initialization (so that any parameter can be initialized).
    The input will be a tensor with at least one dimension. The last dimension
    will contain the vectors. The output is a tensor that will have the same
    size as the input.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        pass

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        raise NotImplementedError("Regularizer not implemented for this operator")

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        return embs.abs()


OPERATORS = PluginRegistry[AbstractOperator]()




class AbstractDynamicOperator(nn.Module, ABC):
    """Perform different operations on many vectors.

    The inputs are a tensor containing a set of vectors and another tensor
    specifying, for each vector, which operation to apply to it. The output has
    the same size as the first input and contains the outputs of the operations
    applied to the input vectors. The different operations are identified by
    integers in a [0, N) range. They are all of the same type (say, translation)
    but each one has its own set of parameters. The dimension of the vectors and
    the total number of operations that need to be supported are provided at
    initialization. The first tensor can have any number of dimensions (>= 1).

    """

    def __init__(self, dim: int, num_operations: int):
        super().__init__()
        self.dim = dim
        self.num_operations = num_operations

    @abstractmethod
    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        pass

    def get_operator_params_for_reg(
        self, operator_idxs: LongTensorType
    ) -> Optional[FloatTensorType]:
        raise NotImplementedError("Regularizer not implemented for this operator")

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        return embs.abs()


DYNAMIC_OPERATORS = PluginRegistry[AbstractDynamicOperator]()

@OPERATORS.register_as("none")
class IdentityOperator(AbstractOperator):
    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return None


@OPERATORS.register_as("diagonal")
class DiagonalOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.diagonal = nn.Parameter(torch.ones((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return self.diagonal.to(embeddings.device) * embeddings

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.diagonal.abs()


@OPERATORS.register_as("translation")
class TranslationOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings + self.translation.to(embeddings.device)

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.translation.abs()


@OPERATORS.register_as("linear")
class LinearOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return torch.matmul(
            self.linear_transformation.to(embeddings.device),
            embeddings.unsqueeze(-1),
        ).squeeze(-1)


@OPERATORS.register_as("affine")
class AffineOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return (
            torch.matmul(
                self.linear_transformation.to(embeddings.device),
                embeddings.unsqueeze(-1),
            ).squeeze(-1) + self.translation.to(embeddings.device)
        )


@OPERATORS.register_as("complex_diagonal")
class ComplexDiagonalOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        assert dim % 2 == 0, "Even dim required"
        self.real = nn.Parameter(torch.ones((dim // 2,)))
        self.imag = nn.Parameter(torch.zeros((dim // 2,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        real_a, imag_a = embeddings[..., : self.dim // 2], embeddings[..., self.dim // 2:]
        real_b, imag_b = self.real.to(embeddings.device), self.imag.to(embeddings.device)
        prod = torch.empty_like(embeddings)
        prod[..., : self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2:] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return torch.sqrt(self.real ** 2 + self.imag ** 2)

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        real, imag = embs[..., : self.dim // 2], embs[..., self.dim // 2:]
        return torch.sqrt(real ** 2 + imag ** 2)


@DYNAMIC_OPERATORS.register_as("none")
class IdentityDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings

    def get_operator_params_for_reg(self, operator_idxs: LongTensorType) -> Optional[FloatTensorType]:
        return None


@DYNAMIC_OPERATORS.register_as("diagonal")
class DiagonalDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.diagonals = nn.Parameter(torch.ones((num_operations, dim)))

    def forward(self, embeddings, operator_idxs):
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return self.diagonals.to(embeddings.device)[operator_idxs] * embeddings

    def get_operator_params_for_reg(self, operator_idxs):
        return self.diagonals[operator_idxs].abs()


@DYNAMIC_OPERATORS.register_as("translation")
class TranslationDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.translations = nn.Parameter(torch.zeros((num_operations, dim)))

    def forward(self, embeddings, operator_idxs):
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings + self.translations.to(embeddings.device)[operator_idxs]

    def get_operator_params_for_reg(self, operator_idxs):
        return self.translations[operator_idxs].abs()


@DYNAMIC_OPERATORS.register_as("linear")
class LinearDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linears = nn.Parameter(torch.eye(dim).repeat(num_operations, 1, 1))

    def forward(self, embeddings, operator_idxs):
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        emb_exp = embeddings.unsqueeze(-1)
        return torch.matmul(
            self.linears.to(embeddings.device)[operator_idxs], emb_exp
        ).squeeze(-1)


@DYNAMIC_OPERATORS.register_as("affine")
class AffineDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linears = nn.Parameter(torch.eye(dim).repeat(num_operations, 1, 1))
        self.translations = nn.Parameter(torch.zeros((num_operations, dim)))

    def forward(self, embeddings, operator_idxs):
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        emb_exp = embeddings.unsqueeze(-1)
        output = torch.matmul(
            self.linears.to(embeddings.device)[operator_idxs], emb_exp
        ).squeeze(-1)
        return output + self.translations.to(embeddings.device)[operator_idxs]


@DYNAMIC_OPERATORS.register_as("complex_diagonal")
class ComplexDiagonalDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        if dim % 2 != 0:
            raise ValueError("Even dimension required.")
        self.real = nn.Parameter(torch.ones((num_operations, dim // 2)))
        self.imag = nn.Parameter(torch.zeros((num_operations, dim // 2)))

    def forward(self, embeddings, operator_idxs):
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        real_a = embeddings[..., : self.dim // 2]
        imag_a = embeddings[..., self.dim // 2:]
        real_b = self.real.to(embeddings.device)[operator_idxs]
        imag_b = self.imag.to(embeddings.device)[operator_idxs]
        prod = torch.empty_like(embeddings)
        prod[..., : self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2:] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self, operator_idxs):
        return torch.sqrt(self.real[operator_idxs] ** 2 + self.imag[operator_idxs] ** 2)

    def prepare_embs_for_reg(self, embs):
        real, imag = embs[..., : self.dim // 2], embs[..., self.dim // 2:]
        return torch.sqrt(real ** 2 + imag ** 2)

# @DYNAMIC_OPERATORS.register_as("complex_diagonal")
# class ComplexDiagonalDynamicOperator(AbstractDynamicOperator):
#     def __init__(self, dim: int, num_operations: int):
#         super().__init__(dim, num_operations)
#         if dim % 2 != 0:
#             raise ValueError(
#                 "Need even dimension as 1st half is real "
#                 "and 2nd half is imaginary coordinates"
#             )
#         self.real = nn.Parameter(torch.ones((num_operations, dim // 2)))
#         self.imag = nn.Parameter(torch.zeros((num_operations, dim // 2)))

#     def forward(
#         self, embeddings: FloatTensorType, operator_idxs: LongTensorType
#     ) -> FloatTensorType:
#         match_shape(embeddings, ..., self.dim)
#         match_shape(operator_idxs, *embeddings.size()[:-1])
#         real_a = embeddings[..., : self.dim // 2]
#         imag_a = embeddings[..., self.dim // 2 :]
#         real_b = self.real.to(embeddings.device)[operator_idxs]
#         imag_b = self.imag.to(embeddings.device)[operator_idxs]
#         prod = torch.empty_like(embeddings)
#         prod[..., : self.dim // 2] = real_a * real_b - imag_a * imag_b
#         prod[..., self.dim // 2 :] = real_a * imag_b + imag_a * real_b
#         return prod

#     def get_operator_params_for_reg(self, operator_idxs: LongTensorType) -> Optional[FloatTensorType]:
#         operator_idxs = operator_idxs.to(self.real.device)
#         return torch.sqrt(self.real[operator_idxs] ** 2 + self.imag[operator_idxs] ** 2)

#     def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
#         assert embs.shape[-1] == self.dim
#         real, imag = embs[..., : self.dim // 2], embs[..., self.dim // 2 :]
#         return torch.sqrt(real ** 2 + imag ** 2)

def instantiate_operator(
    operator: str, side: Side, num_dynamic_rels: int, dim: int
) -> Optional[Union[AbstractOperator, AbstractDynamicOperator]]:
    if num_dynamic_rels > 0:
        dynamic_operator_class = DYNAMIC_OPERATORS.get_class(operator)
        return dynamic_operator_class(dim, num_dynamic_rels)
    elif side is Side.LHS:
        return None
    else:
        operator_class = OPERATORS.get_class(operator)
        return operator_class(dim)