import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.platform import gfile
import numpy as np
import os
import data_reader
from train_model import model


x, y, test_data, vocabulary = data_reader.load_data()

x_train, x_val = np.concatenate((x[:26319],x[:26319],x[:26319],x[:26319],x[:26319],x[:26319],x[:26319],x[:26319],x[:26319],x[:26319], x[26349:-300])), np.concatenate((x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349],x[26319:26349], x[-300:]))
y_train, y_val = np.concatenate((y[:26319],y[:26319],y[:26319],y[:26319],y[:26319],y[:26319],y[:26319],y[:26319],y[:26319],y[:26319], y[26349:-300])), np.concatenate((y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349],y[26319:26349], y[-300:]))
_index = np.random.permutation(np.arange(len(x_train)))
x_train = x_train[_index]
y_train = y_train[_index]

article_length = x_train.shape[1]
out_dir = "logs/"
sess = tf.Session()
with sess.as_default():
  model_ = model(article_length=article_length,vocab_size=len(vocabulary))

  global_step = tf.Variable(0, name="global_step", trainable=False)
  optimizer = tf.train.AdamOptimizer(2e-4)
  train_op = optimizer.apply_gradients(optimizer.compute_gradients(model_.loss), global_step=global_step)
  
  
  saver = tf.train.Saver(tf.all_variables())
  sess.run(tf.initialize_all_variables())
  ckpt = tf.train.get_checkpoint_state(os.path.join(out_dir, 'checkpoints'))

  def eval(x_batch, y_batch):
    feed_dict = {model_.train_articles: x_batch, model_.dropout: 1.0}
    ans1 = sess.run([model_.pred_score], feed_dict)
    auc = roc_auc_score(y_batch,ans1[0])
    if(auc > 0.92):
        test(test_data)
        return True
    print("\n" + str(auc))
    return False
      
  def test(x_batch):
    len_ = x_batch.shape[0]
    output = open("./pre.txt","w")
    output_ = open("./pre_score.txt","w")
    output__ = open("./pre_score1.txt","w")
    for k in range(len_):
      print("running " + str(k))
      feed_dict = {
        model_.train_articles: np.array([x_batch[k]]), model_.dropout: 1.0
      }
     #ans = sess.run([model_.predictions], feed_dict)
      ans1 = sess.run([model_.pred_score], feed_dict)
      #output.write('%f\n' % ans[0])
      #output_.write('%f\n' % ans1[0][0][0])
      output__.write('%f\n' % (1.0 - ans1[0][0][0]))
    print("test_finished!")

  batches = data_reader.batch_iter(zip(x_train, y_train), 16, 10)
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  for batch in batches:
    x_batch, y_batch = zip(*batch)
    feed_dict = {model_.train_articles: x_batch,model_.train_labels: y_batch,model_.dropout: 0.5}
    op, step, loss, accuracy = sess.run([train_op, global_step,  model_.loss, model_.accuracy],feed_dict)
    print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
    current_step = tf.train.global_step(sess, global_step)
    if current_step % 100 == 0:
      if eval(x_val, y_val):
          break
      saver.save(sess, os.path.join(checkpoint_dir, "model"), global_step=current_step)
