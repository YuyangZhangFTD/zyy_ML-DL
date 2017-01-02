# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    this code is for python3
"""


import tensorflow as tf         # import tensorflow

c = tf.constant(1.5)            # creat a constant
x = tf.Variable(1.0, name="x")  # creat a variable

add_op = tf.add(x, c)           # creat add operation
assign_op = tf.assign(x, add_op)# creat assign operation

init = tf.global_variables_initializer()    # init variables    *

sess = tf.Session()             # get session object

sess.run(init)                  # run session

print(sess.run(x))              # should print 1.0

print(sess.run(add_op))         # should print 2.5(x+c=2.5)
print(sess.run(x))              # should print 1.0(x has not been assigned)

print(sess.run(assign_op))      # should print 2.5

print(sess.run(x))              # should print 2.5(x has been assigned)

print(sess.run(x))              # should print out 2.5

sess.close()                    # close session


