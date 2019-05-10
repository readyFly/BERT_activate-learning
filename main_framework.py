"""
#取得训练集数据
def get_train_data():


#取得测试集数据
def get_test_data():


#取得效果评测集数据
def get_test_data():


#更新训练集数据
def update_train_data():


#从未标注池中选择样本
def select_data():

	return selected_data


#更正预测数据（专家标注->给出正确标签) 实际操作可能是找到正确标签
def correct_data():


#评测模型精确度
def test_score():


#调用训练模型
def bert_model():


def fit_model(train_data):

	return new_model
"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
digits = load_digits()
X_data = digits.data.astype(np.float32)
Y_data = digits.target.reshape(-1,1).astype(np.float32)

print(X_data.shape)
print(Y_data.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_data = scaler.fit_transform(X_data)
from sklearn.preprocessing import OneHotEncoder
Y = OneHotEncoder().fit_transform(Y_data).todense() #one-hot编码
print(Y)
batch_size = 10 # 使用MBGD算法，设定batch_size为10
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end, :]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch

tf.reset_default_graph()
tf_X = tf.placeholder(tf.float32,[None,64])
tf_Y = tf.placeholder(tf.float32,[None,10])

tf_W_L1 = tf.Variable(tf.zeros([64,10]))
tf_b_L1 = tf.Variable(tf.zeros([1,10]))

pred = tf.nn.softmax(tf.matmul(tf_X,tf_W_L1)+tf_b_L1)
loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred,1e-11,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) #优化

y_pred = tf.argmax(pred,1)
bool_pred = tf.equal(tf.argmax(tf_Y,1),y_pred)

accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # 准确率

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(2001): # 迭代2001个周期
        for batch_xs,batch_ys in generatebatch(X_data,Y,Y.shape[0],batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={tf_X:batch_xs,tf_Y:batch_ys})
        if(epoch%1000==0):
            res,prediction = sess.run([accuracy,pred],feed_dict={tf_X:X_data,tf_Y:Y})
    res_ypred = y_pred.eval(feed_dict={tf_X:X_data,tf_Y:Y}).flatten()

    print(res_ypred)

from sklearn.metrics import  accuracy_score
print(accuracy_score(Y_data,res_ypred.reshape(-1,1)))
index_true = []
for i in range(len(Y_data)):
	if Y_data[i] != res_ypred.reshape(-1,1)[i]:
		index_true.append(i)
		print(prediction[i])
		print(Y_data[i])
		print(res_ypred[i])

index_error = []
for i in range(len(prediction)):
	p = prediction[i]
	max_n = max(p)
	max_index = np.argmax(p)
	p = np.delete(p,max_index)
	sen_max = max(p)

	diff = max_n - sen_max

	if diff < 0.5:
		index_error.append(i)

print(index_true,index_error)