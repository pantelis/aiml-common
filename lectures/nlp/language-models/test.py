# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pantelis-nlp/tutorial-nlp-notebooks/blob/main/rnn_language_model.ipynb)
# 
# # CNN Language Model
# 
# The following was developed by Harini Appansrinivasan, NYU as part of an assignment submission. 

# %%
import numpy as np
from numpy import array
import keras as K
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout

# split input sequence into inputs X of size = n_steps and targets Y with the subsequent integer after each X
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps                                     # find the end of this pattern
		if end_ix > len(sequence)-1:                             # condition to check if we are beyond the input sequence
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]      # gather input and output parts of the pattern
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

data = 'Chios island is crescent or kidney shaped, 50 km (31 mi) long from north to south, and 29 km (18 mi) at its widest, covering an area of 842.289 km2 (325.210 sq mi).[2] The terrain is mountainous and arid, with a ridge of mountains running the length of the island. The two largest of these mountains, Pelineon (1,297 m (4,255 ft)) and Epos (1,188 m (3,898 ft)), are situated in the north of the island. The center of the island is divided between east and west by a range of smaller peaks, known as Provatas.'

# creating a vocabulary of unique characters
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

# creating a dictionary, mapping characters to index and index to characters
char_to_ix = { ch:i for i,ch in enumerate(chars) }
print(char_to_ix)
ix_to_char = { i:ch for i,ch in enumerate(chars) }
print(ix_to_char)
seq_length = 25        # number of characters in each input sequence from which the next char is predicted
data_tokens = [char_to_ix[ch] for ch in list(data)]               # convert all characters in data string into t tokens
X_tokens, y_tokens = split_sequence(data_tokens, seq_length)      # split into samples, inputs X and targets y
for i in range(len(X_tokens)):   
	print(X_tokens[i], y_tokens[i])                                 # summarize the split input and target tokens
num_samples = X_tokens.shape[0]
print(X_tokens.shape)     # shape of tokenized input array with sequence length = 25
print(y_tokens.shape)     # shape of tokenized target array
print(num_samples)        # total number of inputs to the CNN
print(seq_length)         # number of characters in each input sequence from which the next char is predicted
print(vocab_size)         # total number of unique characters, ie, number of output classes
# function to one-hot encode the data for each unique class k=vocab_size
def onehot_encoding(data):
  if (len(data.shape)==2):
    onehot_data = np.zeros((data.shape[0] * data.shape[1], data.max()+1), dtype=int)   # a 3D array of 0's
    onehot_data[np.arange(data.shape[0] * data.shape[1]), data.flatten()] = 1          # replace 0 with 1 at that index of the original array
    onehot_data = onehot_data.reshape(data.shape[0], data.shape[1], vocab_size)        # reshape into [num_samples, seq_length, vocab_size] format
  elif (len(data.shape)==1):
    onehot_data = np.zeros((data.shape[0], data.max()+1), dtype=int)                   # a 2D array of 0's
    onehot_data[np.arange(data.shape[0]), data.flatten()] = 1                          # replace 0 with 1 at that index of the original array
    onehot_data = onehot_data.reshape(data.shape[0], vocab_size)                       # reshape into [num_samples, vocab_size] format
  return onehot_data
X_onehot = onehot_encoding(X_tokens)  # one-hot encode the inputs
X_onehot.shape                        # shape = [num_samples, seq_length, vocab_size]
y_onehot = onehot_encoding(y_tokens)  # one-hot encode the targets
y_onehot.shape                        # shape = [num_samples, vocab_size]


# %%
tf.keras.backend.clear_session

# define model
model = Sequential()
# conv1d layer with input dim (seq_length, vocab_size), ReLU activation, 64 7x7 filters
model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(seq_length, vocab_size)))  
# maxpool layer, with pool size=2
model.add(MaxPooling1D(pool_size=2))
# flatten the data before passing it to a fully connected / dense layer
model.add(Flatten())
# a dense layer with output size=43 for the 43 unique classes and a softmax activation to give the 43 class probabilities
model.add(Dense(43, activation='softmax'))
# compile the model with adam optimizer and cross entropy loss. Print accuracy metrics while training the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()



# %%
X_input = X_onehot
y_target = y_onehot

# function to test the model and predict characters after each epoch
class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    txt = ''
   
    x = X_input[0:1,:,:]                          # set the first input as the first 25 characters
    for i in range(seq_length):                   # convert the one_hot encoded x inputs to tokens and then to characters, add them to string 'txt'
      txt = ''.join([txt,ix_to_char[np.argmax(x[0,i,:])]])

    n = 0                                         # data pointer
    while(n < num_samples):                       # run the prediction loop until the end of total number of inputs to the CNN
      n = n + 1            
      y_pred = self.model.predict(x, verbose=0)   # predict the class probabilities for the next character from a sequence of 25 encoded characters
      ix = np.argmax(y_pred)                      # find the index of the value with the largest probability
      txt = ''.join([txt,ix_to_char[ix]])         # convert the prediction into character and add it to the list of previous predictions 

      # at test-time, feed back the predicted character to model for next character prediction
      ypred_onehot = np.zeros((1,vocab_size))     # one-hot encode the predicted probabilities
      ypred_onehot[:,ix] = 1

      x = x.reshape(seq_length,vocab_size)        # reshape x into [seq_length, vocab_size] before stacking y_pred
      x = np.vstack([x[1:,:],ypred_onehot])       # remove the first character from x and stack y_pred for the next iteration
      x = x.astype(int)                           # numpy vstack returns a float array, convert it into an integer array
      x = x.reshape(1,seq_length,vocab_size)      # reshape the input x into [1, seq_length, vocab_size]

    print('----\n %s \n----' % (txt, ))           # print the entire string of our model predictions



# %%
# train the model for 100 epochs on X_input
# call the PredictionCallback() function to predict all characters and print them after each epoch
# print the loss and accuracies after each epoch
hist = model.fit(X_input, y_target, epochs=100, verbose=2, callbacks=[PredictionCallback()])


