# Implementing Attention Layer

# Import necessary functions
import tensorflow as tf

class AttentionLayer(tf.keras.Model):
    """
    A class that implements the attention mechanism using the
    following formula:

        attention coefficient = alpha = W.T * f(W1*values + W2*Query)
        Attention_based_values = alpha * values
    """
    def __init__(self, attn_unit: int, **kwargs):
        """
         A class that implements the attention mechanism using the
        following formula:

            attention_map = alpha = W.T * f(W1*values + W2*Query)
            Attention_based_values = alpha * values

        :param attn_unit: The units of attention to apply - Hyper-parameter
        """
        super(AttentionLayer, self).__init__(**kwargs)

        self.attn_unit = attn_unit
        self.W = tf.keras.layers.Conv2D(1, 1, activation = "sigmoid", padding = "same")
        self.W1 = tf.keras.layers.Conv2D(attn_unit, 1, activation = "linear", padding = "same")
        self.W2 = tf.keras.layers.Conv2D(attn_unit, 1, activation = "linear", padding = "same")

    def call(self, query, values, **kwargs):
        """
        Function to do a single forward pass with the
        attention mechanism

        :param query: The sequence of query images to use
        for picking the relevant information from the values
        :param values: The sequence of values to pick from for
        the attention mechanism using query

        :return: A tuple of attention_based_values, attention_map
        """
        attention_map = self.W(tf.nn.relu(self.W1(query) + self.W2(values)))
        attention_based_values = attention_map * values

        return attention_based_values, attention_map

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            "attn_unit": self.attn_unit
        })
        return config

