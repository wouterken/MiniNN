require_relative '../mininn'
require 'pry'

nn = MiniNN::NN.new(inputs: {v1: 100, v2: 100, v3:100}, output: {square: 100}, layers: [6,6,6,6])

nn.train 50, M[
  [1,1, -1],
  [2,2, 0],
  [3,3, 3],
  [4,4, 8],
  [5,5, 15],
  [6,6, 24]
], M[
  [1],
  [4],
  [9],
  [16],
  [25],
  [36]
]



binding.pry

puts nn