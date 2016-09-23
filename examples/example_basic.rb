require_relative '../mininn'
require 'pry'

nn = MiniNN::NN.new(inputs: {sleep: 24, study: 24}, outputs: {score: 100, happiness: 10}, layers: [30,40,30])

nn.train 50_000, \
      M[[3, 5],
       [5, 1],
       [10, 2]],

       M[[75, 8],
        [82, 9],
        [93, 10]]

puts nn.solve([3,4])
puts nn.solve([3,5])
puts nn.solve([10,4])
puts nn.solve([10,1])
puts nn.solve([10,2])
