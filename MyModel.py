
# setting python working environment ...

# import tensorflow as tf
# from tf import keras 

import keras
from keras.models import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D



def create_model(max_len, n_words, n_tags, pos_tags , max_len_chars, n_chars):


	word_input = Input(shape=(max_len,))
	word_emb = Embedding(input_dim=n_words + 2, output_dim=20,input_length=max_len, mask_zero = True)(word_input)

	print(word_emb)
	char_input = Input(shape =(max_len, max_len_chars))
	char_emb = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim = 10,input_length= max_len_chars, mask_zero= True))(char_input)
	print(char_emb)
	char_enc = TimeDistributed(LSTM(units = 20, return_sequences=False,recurrent_dropout = 0.6 ))(char_emb)
	print(char_enc)
	x = keras.layers.concatenate([word_emb, char_enc])
	x = SpatialDropout1D(0.3)(x)
	print(x)
	main_lstm = Bidirectional(LSTM(units = 50, return_sequences = True, 
							 recurrent_dropout = 0.6))(x)
	print(main_lstm)
	out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)
	print(out)
	# print(concat_in)
	# print("\nword_input:",word_emb)
	# # print("\npos_input:",pos_input)

	# model = Dropout(0.1)(word_emb)
	# model = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout= 0.1))(model)
	# # Dense Layer with 50 neurons
	# model = TimeDistributed(Dense(50, activation = "relu"))(model)
	# crf = CRF(n_tags)
	# # output layer is crf
	# out = crf(model)

	# model creation for compilation
	# model = Model(inputs = [word_input,pos_input] , outputs =out)
	# model = Model([word_in, char_in], out)

	model = Model(inputs = [word_input, char_input] , outputs =out)
	# model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
	print(model)
	model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy", metrics = ["acc"])
	return model


# utils 
def plot_history(history):

	import pandas as pd
	hist = pd.DataFrame(history.history)
	plt.figure(figsize=(12,12))
	plt.plot(hist["acc"])
	plt.plot(hist["val_acc"])
	plt.legend()
	plt.show()

def draw_histogram(sentences):
	
	plt.style.use("ggplot")
	plt.hist([len(s) for s in sentences ], bins = 50)
	plt.show()
