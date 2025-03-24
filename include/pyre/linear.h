#ifndef PYRE_LINEAR_H
#define PYRE_LINEAR_H

#include "ember/tensor.h"

namespace pyre {

/**
 * `Linear` represents a densely connected layer in a neural network.
 *
 *
 * `Linear` performs a linear operation on the input tensor. The operation is
 * defined as: $output = inputs * weights + bias$, where each of those terms
 * are tensors with the following shape:
 *
 * - inputs:   num_samples  x num_features
 * - weights:  num_features x num_neurons
 * - bias:     1            x num_neurons
 * - output:   num_samples  x num_neurons
 */
class Linear {
private:
  // The number of features of the inputs to this layer.
  std::size_t num_features_in;
  // The number of neurons that collectively make up this layer.
  std::size_t num_features_out;
  // The tensor representing the weights of this layer. It has a shape of
  // (num_features_in x num_neurons)
  ember::Tensor weights;
  // The inherent activation of this neuron, it is added the the computed
  // activation to determine the final activation of each neuron.
  ember::Tensor bias;

public:
  Linear(std::size_t num_features_in, std::size_t num_features_out);

  /**
   * Performs a linear operation on the input tensor.
   *
   * @param input The input tensor to the layer.
   * @return The output tensor from the layer.
   */
  ember::Tensor operator()(ember::Tensor input);

};  // struct Linear

}  // namespace pyre

#endif  // !PYRE_LINEAR_H
