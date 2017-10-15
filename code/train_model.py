import tensorflow as tf

class model:
  def __init__(self, article_length, vocab_size, emb_len = 128, num_classes = 2):
      
      self.train_articles = tf.placeholder(tf.int32, [None, article_length])
      self.train_labels = tf.placeholder(tf.float32, [None, num_classes])
      self.dropout = tf.placeholder(tf.float32)
      
      W = tf.Variable(tf.random_normal([vocab_size, emb_len],mean=-1.0,stddev=1.0))
      self.embedded = tf.nn.embedding_lookup(W, self.train_articles)
      highway1 = []
      filters = [1,2,3,4,5,6,8,1,2,3,4]
      filters_num = [100,150,150,200,200,200,200,160,160,160,160]
      for pr in range(11):
        fs = filters[pr]
        fn = filters_num[pr]
        W = tf.Variable(tf.random_normal([fs, emb_len, 1, fn], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[fn]))
        conv = tf.nn.conv2d(tf.expand_dims(self.embedded, -1),W,strides=[1, 1, 1, 1],padding="VALID")
        conv = tf.nn.relu(tf.nn.bias_add(conv, b))
        print(conv)

        if fn in [160]:
            W = tf.Variable(tf.random_normal([fs, 1, fn, fn*2], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[fn*2]))
            conv = tf.nn.conv2d(conv,W,strides=[1, 1, 1, 1],padding="SAME")
            conv = tf.nn.relu(tf.nn.bias_add(conv, b))
        conv = tf.nn.max_pool(conv,ksize=[1, conv.get_shape()[1], 1, 1],strides=[1, 1, 1, 1],padding='VALID')
        highway1.append(conv)

      highway1 = tf.reshape(tf.concat(3, highway1), [-1, sum(filters_num) + 4*160])
      h = highway1
      h = tf.nn.relu(tf.nn.rnn_cell._linear(h, highway1.get_shape()[1], 0))
      x = tf.sigmoid(tf.nn.rnn_cell._linear(highway1, highway1.get_shape()[1], 0))
      highway1 = x * highway1 + (1.0 - x) * highway1
      highway1 = tf.nn.dropout(highway1, self.dropout)
      highway2 = highway1
      W = tf.Variable(tf.random_normal([-1, sum(filters_num) + 4*160, num_classes], stddev=0.1))
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
      self.scores = tf.nn.xw_plus_b(self.highway1, W, b)

      self.pred_score = tf.nn.softmax(self.scores)
      train_loss = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.train_labels)
      self.loss = tf.reduce_mean(train_loss)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.train_labels, 1)), "float"))
