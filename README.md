## MiniNN
A Toy neaural network written in Ruby

Construct a small NN with an input layer (with named inputs), an array describing the number of hidden layers and their sizes, and an output layer (with labels)

### Example
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

### Binary ops

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