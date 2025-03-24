#include "pyre/linear.h"

using namespace pyre;


Linear::Linear(std::size_t num_features_in, std::size_t num_features_out)
    : num_features_in(num_features_in), num_features_out(num_features_out) {
  weights = ember::Tensor::randn({num_features_in, num_features_out});
  bias = ember::Tensor::randn({num_features_out});
}

ember::Tensor Linear::operator()(ember::Tensor input) {
  return input * weights + bias;
}
