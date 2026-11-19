func.func @add(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ttir.add %a, %b : tensor<16xf32>
  return %0 : tensor<16xf32>
}
