require 'rmatrix'

module MiniNN
  class NN
    attr_accessor :inputs, :outputs, :layers, :synapses

    def initialize(inputs:, outputs:, layers:)
      self.inputs   = named_struct(inputs)
      self.outputs  = named_struct(outputs)
      self.synapses = ([inputs.size] + layers + [outputs.size]).each_cons(2).map do |rows, cols|
        2 * M.blank(rows: rows, columns: cols).random! - 1
      end
    end

    def named_struct(params)
      Struct.new(*params.keys).new(*params.values)
    end

    def scale_inputs(set)
      set / inputs.values
    end

    def scale_result_inputs(results)
      results / outputs.values
    end

    def scale_result_outputs(results)
      results * outputs.values
    end

    def sigmoid(x)
      1.0 / ( 1 + Math::E ** -x)
    end

    def derivative(x)
      x.mult(1 - x)
    end

    def train(*iterations, training_set, training_results)
      training_set     = scale_inputs(M[training_set])
      training_results = scale_result_inputs(M[training_results])

      (iterations.first || 500_000).times do |j|
        current_layer = training_set

        layers = [current_layer] + synapses.map do |synapse|
          current_layer = sigmoid(current_layer * synapse)
        end

        error = training_results - layers.last
        print "\rError: #{error.narray.abs.mean}" if j % 100 == 0

        layers[1..-1].zip(synapses).reverse.map do |layer, synapse|
          delta = error.mult derivative(layer)
          error = delta * synapse.T
          delta
        end.reverse.zip(synapses, layers[0..-2]).each_with_index do |(delta, synapse, layer), index|
          synapses[index] = synapse + layer.T * delta
        end

      end
    end

    def solve(inputs, dp: 4)
      values = scale_inputs(M[inputs])
      results = scale_result_outputs(([values] + synapses.map do |syn|
        values = sigmoid(values * syn)
      end).last).round(dp).to_a
      outputs.members.zip(results).to_h
    end
  end
end