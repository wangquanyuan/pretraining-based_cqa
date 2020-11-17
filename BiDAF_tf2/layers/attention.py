import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):
    def __int__(self, ):
        super(C2QAttention, self).__init__()

    def call(self, similarity, qencode):
        qencode = tf.expand_dims(qencode, axis=1)
        att = tf.keras.activations.softmax(similarity, axis=-1)
        c2q = tf.expand_dims(att, axis=-1)
        c2q_att = tf.math.reduce_sum(c2q * qencode, axis=-2)

        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(Q2CAttention,self).__init__()

    def call(self, similarity, cencode):
        max_similarity = tf.math.reduce_max(similarity, axis=-1)
        att = tf.keras.activations.softmax(max_similarity)
        q2c = tf.expand_dims(att,axis=-1)
        weigthed_sum = tf.math.reduce_sum(q2c * cencode, axis=-2)
        expanded_weighted_sum = tf.expand_dims(weigthed_sum, axis=1)
        num_of_repeatations = cencode.shape[1]
        q2c_att = tf.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

        return q2c_att