# import necessary python packages
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth = 1.0):
    """
    Function to implement the dice co-efficient for the
    given output segmentation map and true segmentation map

    :param y_true: Tensor of shape (batch_size, width, height) consisting
    of the true map
    :param y_pred: Tensor of shape (batch_size, width, height) consisting
    of the predicted map
    :param smooth: this parameter is very important as it reduces the
    overfitting. This is called as Laplacian smoothing and additive
    smoothing

    :return: the dice coef derived from the following formula

        dice(A, B) = (2 * sum(A.B) + smooth) / (sum(A) + sum(B) + smooth)
    """
    # y_pred =  y_pred[:, :, :, -1]
    # y_true = y_true[:, :, :, -1]

    true_flat = tf.abs(tf.reshape(y_true, shape = [-1], name = "y_true_flat"))
    pred_flat = tf.abs(tf.reshape(y_pred, shape = [-1], name = "y_pred_flat"))

    num = 2 * tf.reduce_sum(tf.multiply(true_flat, pred_flat)) + smooth
    den = tf.reduce_sum(true_flat) + tf.reduce_sum(pred_flat) + smooth

    return num / den

def dice_loss(y_true, y_pred, smooth = 1.0):
    """
    Function to implement the dice loss for the
    given output segmentation map and true segmentation map

    :param y_true: Tensor of shape (batch_size, width, height) consisting
    of the true map
    :param y_pred: Tensor of shape (batch_size, width, height) consisting
    of the predicted map
    :param smooth: this parameter is very important as it reduces the
    overfitting. This is called as Laplacian smoothing and additive
    smoothing

    :return: the dice loss derived from the following formula

        dice(A, B) = 1.0 - (2 * sum(A.B) + smooth) / (sum(A) + sum(B) + smooth)
    """
    return 1.0 - dice_coef(y_true, y_pred, smooth)

# def dice_coef_K(y_true, y_pred, smooth=1):
# #     """
# #     Dice = (2*|X & Y|)/ (|X|+ |Y|)
# #          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
# #     ref: https://arxiv.org/pdf/1606.04797v1.pdf
# #     """
# #     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
# #     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
# #
# # def dice_coef_loss_K(y_true, y_pred):
# #     return 1-dice_coef(y_true, y_pred)