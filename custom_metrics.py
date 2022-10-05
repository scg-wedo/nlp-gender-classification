import tensorflow as tf

def category_acc(y_true, y_pred):
    gt = tf.math.argmax(y_true, axis=1)
    prediction = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(prediction, gt)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy