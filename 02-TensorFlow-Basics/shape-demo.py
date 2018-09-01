import tensorflow as tf

# tf.InteractiveSession()

# Rank 0 : Scalar (magnitude only)
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

init= tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init) # 必须得先运行
    s.run([mammal, ignition, floating, its_complicated])
    print('mammal', mammal, mammal.eval())
    print('ignition', ignition, ignition.eval())
    print('floating', floating, floating.eval())
    print('its_complicated', its_complicated, its_complicated.eval())

# Rank 1: Vector (magnitude and direction)

mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

init= tf.global_variables_initializer()

print('==============')

with tf.Session() as s:
    s.run(init) # 必须得先运行
    s.run([mystr, cool_numbers, first_primes, its_complicated])
    print('mystr', mystr, mystr.eval())
    print('ignition', cool_numbers, cool_numbers.eval())
    print('first_primes', first_primes, first_primes.eval())
    print('its_very_complicated', its_very_complicated, its_very_complicated.eval())

 # Rank 3 : Matrix (table of numbers)
 
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)

init= tf.global_variables_initializer()

print('==============')

with tf.Session() as s:
    s.run(init) # 必须得先运行
    s.run([mymat, linear_squares])
    print('mymat', mymat, mymat.eval())
    print('linear_squares', linear_squares, linear_squares.eval())
    print('squarish_squares', squarish_squares, squarish_squares.eval())

# shape 的含义：第一个：一共有几根，第二个：每个里面有几个
    