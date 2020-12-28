import tensorflow as tf
input_x = [
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
]
print(input_x)
with tf.Session() as sess:
    input_x=tf.transpose(input_x,(0,2,1))
    print(sess.run(input_x))