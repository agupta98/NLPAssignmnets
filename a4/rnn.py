"""
See README.md
"""


import tensorflow as tf
import pandas as pd

#to read data from train.txt file
words = open('train.txt','r').readlines()
print(len(words))
li1=[]
li2=[]

#to extract data
words = [x.strip('\n') for x in words]
for i in words:
    if (i=='-DOCSTART- -X- -X- O'):
        continue
    if (i !=''):
        i = i.split(' ')
        li1.append(tuple(i))
    if (i ==''):
        li2.append(li1)
        li1=[]
li2.append(li1)
li1=[]
data=list(filter(None,li2))

#to form list of lists comprising of sentence
li_sentences = []
for sentence in data:
    li_temp = []
    for word in sentence:
        li_temp.append(word[0])
    li_sentences.append(li_temp)
li_sentences.append(li_temp)

#to form list of lists of entities
li_namedEntity = []
for sentence in data:
    li_temp = []
    for word in sentence:
        li_temp.append(word[-1])
    li_namedEntity.append(li_temp)
li_namedEntity.append(li_temp)

#to form list of tokens
li_tokens =[j for i in li_sentences for j in i]
#to form list of entity
li_net = [j for i in li_namedEntity for j in i]

#to form pandas dataframe tokens and corresponding named entity
df = pd.DataFrame({'CL':li_net,'Tokens':li_tokens})

#One hot Encode the tokens in the dataframe
df_features = pd.get_dummies(pd.Series(li_tokens))
df_features
#One hot Encode the labels
df_labels = pd.get_dummies(pd.Series(li_net))
df_labels
#Data Preparation to feed data in Tensor flow End


learning_rate = 0.001
training_epochs = 15

n_input = 6856
#nuber of steps that are divisible by 27876
batch_size = 12
n_classes = 5
n_hidden = 20
n_steps = 12

weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden])),
          'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
          }

biases = {'b1':tf.Variable(tf.random_normal([n_hidden,])),
          'out':tf.Variable(tf.random_normal([n_classes,]))
          }
# tf Graph Input
x = tf.placeholder(tf.float32,[None,n_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
sess = tf.Session()



#df_features = df_features.as_matrix()
#df_labels = df_labels.as_matrix()
#data_tensor = tf.constant(df_features,dtype = tf.float32,shape=[27876, 6856])
#label_tensor = tf.constant(df_labels, 'float32',shape=[27876, 5])

def rnn(x,weights,biases):
    x = tf.unstack(x, n_steps, 1)
    x = tf.reshape(x, [-1, n_input])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1, states = tf.nn.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    out_layer = tf.matmul(layer_1,weights['out']) + biases['out']
    return out_layer
    
pred = rnn(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(training_epochs):
    avg_costs =0.0
    total_batch = int(27876/12)
    for i in range(total_batch):
        batch_x = df_features[12*i:12*(i+1)].as_matrix()
        #print(batch_x.shape)
        batch_x = batch_x.reshape([1, n_steps, n_input])
        batch_y = df_labels[12*i:12*(i+1)]
        _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        avg_costs += c / total_batch
    print("%d,%d"%(epoch+1,avg_costs))


