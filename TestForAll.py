import matplotlib.pyplot as plt
import tensorflow as tf

# tf.gfileGFile()函数：读取图像
image_png = tf.gfile.FastGFile(r'C:\Users\BZT\Desktop\Pic-Current\Test.png', 'rb').read()

with tf.Session() as sess:
    image_jpg = tf.image.decode_jpeg(image_png)  # 图像解码
    print(sess.run(image_jpg))  # 打印解码后的图像（即为一个三维矩阵[w,h,3]）
    image_jpg = tf.image.convert_image_dtype(image_jpg, dtype=tf.uint8)  # 改变图像数据类型

    image_png = tf.image.decode_png(image_png)
    print(sess.run(image_jpg))
    image_png = tf.image.convert_image_dtype(image_png, dtype=tf.uint8)

    plt.figure(1)  # 图像显示
    plt.imshow(image_jpg.eval())
    plt.figure(2)
    plt.imshow(image_png.eval())
    plt.show()
