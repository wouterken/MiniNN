require_relative '../mininn'
require 'pry'

nn = MiniNN::NN.new(inputs: {a1: 1, a2: 1, a3: 1}, outputs: {result: 1}, layers:[4, 3])

nn.train [
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
    [0,0,0],
    [1,0,0]
 ],

 V[0,1,1,0,1,0].T

puts nn.solve([0,0,1], dp: 2)