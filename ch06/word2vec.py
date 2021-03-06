import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = "logs/word2vec_intro"


digit_to_word_map ={1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
sentences = []

# 홀수 시퀀스/짝수 시퀀스 두 종류의 문장을 생성
for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))



sentences[0:10]
# ['One Three Three',
#  'Two Two Two',
#  'One One Seven',
#  'Eight Two Four',
#  'Three One Seven',
#  'Eight Two Eight',
#  'Seven Five Five',
#  'Six Four Six',
#  'Five Five Nine',
#  'Two Eight Eight']



# 단어를 인덱스에 매핑
word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)


# 스킵-그램 쌍 생성
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent) - 1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                             word2index_map[tokenized_sent[i]]]
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][1]])


def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y



skip_gram_pairs[0:10]
# [[1, 0],
#  [1, 1],
#  [2, 2],
#  [2, 2],
#  [0, 0],
#  [0, 3],
#  [2, 4],
#  [2, 5],
#  [0, 1],
#  [0, 3]]



# 배치 예제
x_batch, y_batch = get_skipgram_batch(8)
x_batch
# [3, 5, 7, 3, 1, 2, 5, 4]
y_batch
# [[8], [4], [7], [8], [0], [5], [7], [4]]
[index2word_map[word] for word in x_batch]
# ['seven', 'four', 'six', 'seven', 'three', 'two', 'four', 'eight']
[index2word_map[word[0]] for word in y_batch]
# ['nine', 'eight', 'six', 'nine', 'one', 'four', 'six', 'eight']



# 입력 데이터와 레이블
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])



# Embedding lookup table currently only implemented in CPU
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # 본질적으로 룩업 테이블이다.
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    

# 잡음 대비 추정(NCE) 손실 계산을 위한 변수 생성
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_dimension],
                        stddev=1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, inputs = embed,
                   labels = train_labels, num_sampled = negative_samples,
                   num_classes = vocabulary_size))
tf.summary.scalar("NCE_loss", loss)


# 학습률 감소
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                          global_step=global_step,
                                          decay_steps=1000,
                                          decay_rate=0.95,
                                          staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)


# 모든 요약 연산을 병합
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))
            
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # 임베딩을 메타데이터 파일과 연결
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)
    
    tf.global_variables_initializer().run()
    
    for step in range(1000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss,
                                  feed_dict={train_inputs: x_batch,
                                             train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))
    
    # 사용 전 임베딩 정규화
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)


ref_word = normalized_embeddings_matrix[word2index_map["one"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])


# nine
# 0.902964
# three
# 0.876791
# seven
# 0.8175905
# five
# 0.7768844
# two
# 0.0069868937
# four
# -0.009589478
# eight
# -0.058123693
# six
# -0.080495425
