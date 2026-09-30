#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <optional>

void cutlass_scaled_mm(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b,
                       torch::stable::Tensor const& a_scales,
                       torch::stable::Tensor const& b_scales,
                       std::optional<torch::stable::Tensor> const& bias);

STABLE_TORCH_LIBRARY(lab_sm90_contract, ops) {
  ops.def("scaled_mm(Tensor! out, Tensor a, Tensor b, Tensor a_scales, "
          "Tensor b_scales, Tensor? bias) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(lab_sm90_contract, CUDA, ops) {
  ops.impl("scaled_mm", TORCH_BOX(&cutlass_scaled_mm));
}
