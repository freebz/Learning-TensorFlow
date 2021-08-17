import tensorflow as tf

# MNIST 데이터 임포트
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 매개변수 정의
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# 텐서보드 모델 요약을 저장할 위치
LOG_DIR = "logs/RNN_with_summaries"

# 입력과 레이블을 위한 플레이스홀더 생성
_inputs = tf.placeholder(tf.float32,
                         shape=[None, time_steps,element_size],
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')




# 요약을 로깅하는 몇몇 연산을 추가하는 헬퍼 함수(텐서플로 문서에서 발췌)
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 입력 및 은닉 계층에 적용할 가중치와 편향값
with tf.name_scope('rnn_weights'):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)
        

def rnn_step(previous_hidden_state,x):

    current_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, Wh) +
        tf.matmul(x, Wx) + b_rnn)

    return current_hidden_state


# scan 함수로 입력값 처리
# 입력의 형태: (batch_size, time_steps, element_size)
processes_input = tf.transpose(_inputs, perm=[1, 0, 2])
# 변형된 입력의 형태: (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# 시간의 흐름에 따른 상태 벡터 구하기
all_hidden_states = tf.scan(rnn_step,
                            processes_input,
                            initializer=initial_hidden,
                            name='states')


# 출력 계층에 적용할 가중치
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                             mean=0, stddev=.01))
        variable_summaries(Wl)

    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                             mean=0, stddev=.01))
        variable_summaries(bl)


# 상태 벡터에 선형 계층 적용
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # 시간에 따라 반복하면서 모든 RNN 결과에 선형 계층 적용
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # 최종 결과
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)



# RNN 분류

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # RMSPropOptimizer 사용
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))

    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar('accuracy', accuracy)

# 요약을 병합
merged = tf.summary.merge_all()


# 작은 테스트 데이터 생성
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # LOG_DIR에 텐서보드에서 사용할 요약을 기록
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                        graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 28개의 시퀀스를 얻기 위해 각 데이터를 28픽셀의 형태로 변환
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary, _ = sess.run([merged, train_step],
                              feed_dict={_inputs: batch_x, y: batch_y})
        # 요약을 추가
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            acc, loss, = sess.run([accuracy, cross_entropy],
                                  feed_dict={_inputs: batch_x, y: batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
                   "{:.5f}".format(acc))
        if i % 10:
            # MNIST 테스트 이미지에서 정확도를 계산해서 요약에 추가
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={_inputs: test_data,
                                               y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: test_label})
    print("Test Accuracy:", test_acc)



# Iter 0, Minibatch Loss= 2.301893, Training Accuracy= 11.71875
# Iter 1000, Minibatch Loss= 1.194614, Training Accuracy= 54.68750
# Iter 2000, Minibatch Loss= 0.687153, Training Accuracy= 78.12500
# Iter 3000, Minibatch Loss= 0.246742, Training Accuracy= 93.75000
# Iter 4000, Minibatch Loss= 0.166810, Training Accuracy= 97.65625
# Iter 5000, Minibatch Loss= 0.103405, Training Accuracy= 96.87500
# Iter 6000, Minibatch Loss= 0.108679, Training Accuracy= 97.65625
# Iter 7000, Minibatch Loss= 0.059725, Training Accuracy= 98.43750
# Iter 8000, Minibatch Loss= 0.062364, Training Accuracy= 98.43750
# Iter 9000, Minibatch Loss= 0.033335, Training Accuracy= 99.21875
# Test Accuracy: 97.65625
