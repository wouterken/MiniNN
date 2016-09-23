require_relative '../mininn'
require 'pry'

nn = MiniNN::NN.new(inputs: {value: 100}, output: {square: 100}, layers: [6,6,6,6])

nn.train M[
  [1],
  [2],
  [3],
  [4],
  [5],
  [6]
], M[
  [1],
  [4],
  [9],
  [16],
  [25],
  [36]
]


