import tensorflow as tf

def create_matrix_summary(matrix, n_nodes):
    """Creates a tf.summary.scalar of each weight in the matrix(layer).

    Args:
        matrix: The weights tensor object.
        n_nodes: Number of weights that the matrix has.

    Returns:
        A list of all the tf.summary.scalar of each weight in the layer.
    """
    summary = []
    for i in range(n_nodes):
        summary.append(tf.summary.scalar(str(i), matrix[0][i]))
    return summary


def create_layer(name, inputs, n_inputs, n_nodes, activation, dropout, keep_prob=0.75):
    """Creates a new layer for a neural network. The activation (if required) function is RELU.

    Args:
        name: String to be appended to 'hidden_layer_'. In this case, it's a str(number) or 'output'.  Example: 'hidden_layer_1'.
        inputs: Inputs to be multiplied with the weights.
        n_inputs: Number of input nodes.
        n_nodes: Number of the hidden nodes of the layer.
        activation: A boolean, it says if the layer has activation function
        dropout: A boolean, it says if the layer is a dropout layer.
        keep_prob: Probability of keep 'on' a weight on the layer.

    Returns:
        returns 3 variables:
            output: The final layer
            summary: A list with all the summary objects (all the tf.summary.scalar of the layer)
            saved_variables: A dictionary with all the variables to be saved and restored in the Session(). In this case are the weights and biases of the layer.
    """
    summary = []
    saved_variables = {}
    with tf.variable_scope("hidden_layer_%s" % name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("weights_%s" % name):
            weights = tf.get_variable(name="weights", shape=[
                n_inputs, n_nodes], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

            saved_variables[weights.name] = weights
            summary += create_matrix_summary(weights, n_nodes)

        with tf.variable_scope("biases_%s" % name):
            biases = tf.get_variable(name="biases", shape=[
                n_nodes], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            saved_variables[biases.name] = biases
        with tf.variable_scope("output_%s" % name):
            output = tf.add(tf.matmul(inputs, weights), biases)
            if activation:
                output = tf.nn.relu(output)
            if dropout:
                output = tf.nn.dropout(output, keep_prob=keep_prob)
            return output, summary, saved_variables


def create_model(inputs, training):
    """Creates a new neural network.

    Args:
        inputs: Specify the inputs for the neural network. Example: tf.placeholder(...)
        training: A boolean. If True: returns the nn with the dropout (if required). Else: returns the nn without dropout.

    Returns:
        Returns 3 variables:
            output_layer: The output layer of the nn.
            summaries: A list with all the tf.summary.scalar of all layers.
            saved_variables: A dictionary with all the variables to be saved.
    """
    n_inputs = 3
    n_outputs = 2
    hidden_layers_nodes = [10, 5]

    summaries = []
    saved_variables = {}

    if training:
        hidden_layer_1, summary_1, saved_variables_1 = create_layer(
            "1", inputs, n_inputs, hidden_layers_nodes[0], True, True, 0.9)
        hidden_layer_2, summary_2, saved_variables_2 = create_layer(
            "2", hidden_layer_1, hidden_layers_nodes[0], hidden_layers_nodes[1], True, True, 0.8)
        output_layer, summary_3, saved_variables_3 = create_layer(
            "output", hidden_layer_2, hidden_layers_nodes[1], n_outputs, False, False)
        summaries += summary_1+summary_2+summary_3

        saved_variables.update(saved_variables_1)
        saved_variables.update(saved_variables_2)
        saved_variables.update(saved_variables_3)

        return output_layer, summaries, saved_variables
    else:
        hidden_layer_1, summary_1, saved_variables_1 = create_layer(
            "1", inputs, n_inputs, hidden_layers_nodes[0], True, False)
        hidden_layer_2, summary_2, saved_variables_2 = create_layer(
            "2", hidden_layer_1, hidden_layers_nodes[0], hidden_layers_nodes[1], True, False)
        output_layer, summary_3, saved_variables_3 = create_layer(
            "output", hidden_layer_2, hidden_layers_nodes[1], n_outputs, False, False)
        summaries += summary_1+summary_2+summary_3

        saved_variables.update(saved_variables_1)
        saved_variables.update(saved_variables_2)
        saved_variables.update(saved_variables_3)

        return output_layer, summaries, saved_variables
