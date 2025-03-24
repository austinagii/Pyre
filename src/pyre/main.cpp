#include "ember/tensor.h"
#include "pyre/linear.h"

#include <iostream>

using namespace pyre;

int main() {
  Module model = Module::Sequential({
    Linear({5, 3}),
    Sigmoid(),
    Linear({3, 5}),
    Sigmoid()
    Linear({5, 1})
  });

  model(ember::Tensor::randn({10, 5}));
  return 0;
}
