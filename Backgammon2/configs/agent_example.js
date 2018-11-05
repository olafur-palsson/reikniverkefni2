// NOTE: this is just an example

const x = {
	// Name of this configuration
	"name": "abc_def",
	// Type of agent (class)
	// * human
	// * random
	// * nn1
	// * best_nn1
	"type": "nn1",

	// Configuration
	"cfg": {

		// * tesauro 
		// * raw1
		// * raw_hot
		"feature_vector": "raw1",

		// Epsilon for epsilon greedy
		"epsilon": 0.15,

		// * net: basic network for testing
		// 
		"net": "basic",

		// How many nodes are in the input layer
		"input_layer": 464,

		// How many hidden layers there are, and how many
		// nodes are in each
		"hidden_layers": [250, 250],

		// How many nodes are in the output layer
		"output_layer": 1,

		// Not sure what this is, but OK.
		// I guess this is for the n-step TD.
		"temporal_delay": 3,

		// Stochastic gradient descent
		"sgd": {
			"momentum": 0.9,
			"learning_rate": 0.00005
		}
	}
}

