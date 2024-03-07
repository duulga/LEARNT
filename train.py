"""
 <Assembly, IR> Mapping unit
 Author: Duulga Baasantogtokh
 Date: 24/01/2024

 usage: change instance_amount then execute

 """

from OptimalMapper import output_emitter
from Tokenizer import tokenizer, vectorizer, normalizer, vectorizer_with_dic
import pickle
import random
import tensorflow as tf
import numpy as np
from posenc import PositionalEmbedding
import matplotlib.pyplot as plt
from transformer import transformer, masked_accuracy, masked_loss, CustomSchedule

CORPUS_SIZE = 25000
ASM_VOCAB_SIZE = 0
IR_VOCAB_SIZE = 0
MAX_BB_LENGTH = 150
ASM_DIC = {}
IR_DIC = {}

pairs = normalizer(output_emitter(CORPUS_SIZE))
with open("temp.pickle", "wb") as fp:
    pickle.dump(pairs,fp)

#Load normalized basic block pairs
with open("temp.pickle", "rb") as fp:
    bb_pairs = pickle.load(fp)

# train-val-test split of randomized basic block pairs
random.shuffle(bb_pairs)
n_val = int(0.15*len(bb_pairs))
n_train = len(bb_pairs) - 2*n_val
train_pairs = bb_pairs[:n_train]
val_pairs = bb_pairs[n_train: n_train+n_val]
test_pairs = bb_pairs[n_train+n_val:]

# print(f"Checking train_pairs")
# for i in range(50):
#     print(f" {i} : {train_pairs[i]}")

def format_dataset(asm, ir):
    source = {"encoder_inputs": asm, "decoder_inputs": ir[:,:-1]}
    #print(f'DEBUGDEBUGDEBUG {source["decoder_inputs"]}')
    target = ir[:,1:]
    return (source, target)

def make_dataset(pairs, batch_size=64):
    # Create Tensorflow Dataset for the basic block pairs
    global MAX_BB_LENGTH
    # unpack the basic blocks
    asm_tokens, ir_tokens, asm_iv_dict, ir_iv_dict, tokenized_pairs = tokenizer(pairs)
    asm_bbs = [pair[0] for pair in tokenized_pairs]
    ir_bbs = [pair[1] for pair in tokenized_pairs]
    _, asm_vectorized = vectorizer_with_dic(ASM_DIC, asm_bbs, MAX_BB_LENGTH)
    _, ir_vectorized = vectorizer_with_dic(IR_DIC, ir_bbs, MAX_BB_LENGTH + 1)

    print("Checking the vectors!...")
    max = 0
    print(asm_vectorized[0][0])
    for vector in asm_vectorized:
        for token in vector:
            if( int(token) > max):
                max = int(token)
            else:
                pass
    print(f"Biggest int in vectorized asembly is: {max}")
    
    max = 0
    print(ir_vectorized[0][0])
    for vector in ir_vectorized:
        for token in vector:
            if( int(token) > max):
                max = int(token)
            else:
                pass
    print(f"Biggest int in vectorized ir is: {max}")
    
    #Create tensors
    dataset = tf.data.Dataset.from_tensor_slices((asm_vectorized, ir_vectorized))
    return dataset.shuffle(2048).batch(batch_size).map(format_dataset).prefetch(16).cache()

print("Getting the size of vocabs")
asm_tokens, ir_tokens, asm_iv_dict, ir_iv_dict, tokenized_pairs = tokenizer(bb_pairs)
asm_bbs = [pair[0] for pair in tokenized_pairs]
ir_bbs = [pair[1] for pair in tokenized_pairs]
ASM_DIC, _, asm_vectorized = vectorizer(asm_tokens, asm_bbs, MAX_BB_LENGTH)
IR_DIC, _, ir_vectorized = vectorizer(ir_tokens, ir_bbs, MAX_BB_LENGTH + 1)
ASM_VOCAB_SIZE = len(ASM_DIC)
IR_VOCAB_SIZE = len(IR_DIC)

# print("Checking the vectors!...")
# max = 0
# for vector in asm_dict:
#     if(asm_dict[vector] > max):
#         max = asm_dict[vector]

# print(f"The largest integer in assembly vector is {max}")

# max = 0
# for vector in ir_dict:
#     if(ir_dict[vector] > max):
#         max = ir_dict[vector]
# print(f"The largest integer in IR vector is {max}")

print("Making dataset...")
train_ds = make_dataset(train_pairs)
print("Making val_pairs")
val_ds = make_dataset(val_pairs)

#test the dataset
# for inputs, targets in train_ds.take(64):
#     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
#     print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
#     print(f'inputs["encoder_inputs"][0]: {inputs["encoder_inputs"][0]}')
#     #tf.debugging.assert_all_finite(inputs["encoder_inputs"][0], "There is")
#     print(f'inputs["encoder_inputs"][1]: {inputs["encoder_inputs"][1]}')
#     print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
#     print(f'inputs["decoder_inputs"][1]: {inputs["decoder_inputs"][1]}')
#     print(f"targets.shape: {targets.shape}")
#     print(f"targets[0]: {targets[0]}")
# print(f"Checking if input data has nan")
# count = 1
# for inputs, targets in train_ds:
#     # assert not np.any(np.isnan(inputs["encoder_inputs"]))
#     # assert not np.any(np.isnan(inputs["decoder_inputs"]))
#     # assert not np.any(np.isnan(targets))
#     print(count)
#     count = count + 1
#     #tf.debugging.assert_all_finite(inputs["encoder_inputs"], "There is")
#     # tf.debugging.assert_all_finite(inputs["decoder_inputs"], "There is")
#     # tf.debugging.assert_all_finite(inputs["targets"], "There is")
    


# vocab_size_asm = ASM_VOCAB_SIZE
# seq_length = MAX_BB_LENGTH

# test the dataset
# # print("Testing Pos enc...")
# for inputs, targets in train_ds:
#     # print(inputs["encoder_inputs"])
#     embed_en = PositionalEmbedding(seq_length, vocab_size_asm, embed_dim=512)
#     en_emb = embed_en(inputs["encoder_inputs"])
#     dec_emb = embed_en(inputs["decoder_inputs"])
#     print(en_emb.shape)
#     print(en_emb._keras_mask)
#     print(dec_emb.shape)
#     # print(dec_emb._keras_mask)
    # tf.debugging.assert_all_finite(inputs["encoder_inputs"], "There is")

seq_len = MAX_BB_LENGTH
# seq_len = 200
num_layers = 4
num_heads = 8
key_dim = 128
ff_dim = 512
dropout = 0.1
# vocab_size_asm = 20000  
# vocab_size_ir = 30000
vocab_size_asm = ASM_VOCAB_SIZE + 100
vocab_size_ir = IR_VOCAB_SIZE + 100

print(f'Initiating model... with ASM_VOCAB_SIZE: {vocab_size_asm}, IR_VOCAB_SIZE: {vocab_size_ir}, seq_len: {seq_len}')
model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                    vocab_size_asm, vocab_size_ir, dropout)
tf.keras.utils.plot_model(model, "transformer.png",
                          show_shapes=True, show_dtype=True, show_layer_names=True,
                          rankdir='BT', show_layer_activations=True)
lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
print('Compiling model...')
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
model.summary()

#Number of epoch to train for
epochs = 20

#Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # To log when training is stopped
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity.
)

# Checkpoint callback to save the model with the best validation loss
checkpoint_path = "asm-llvm-transformer_checkpoint.h5"  # Path where to save the model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,  # Only save a model if `val_loss` has improved
    verbose=1  # Log when a model is being saved
)

print('Training the model...')
history = model.fit(
    train_ds, 
    epochs=epochs, 
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint])

#Save the trained model
model.save("asm-llvm-transformer.h5")

# Plot the loss and accuracy history
actual_epochs = len(history.history["loss"])
x = list(range(1, actual_epochs + 1))
fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True)
fig.suptitle('Training History')
axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
axs[0].set_ylabel("Loss")
axs[0].legend(loc="upper right")
axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="acc")  # Adjust key if necessary
axs[1].plot(x, history.history["val_masked_accuracy"], alpha=0.5, label="val_acc")  # Adjust key if necessary
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].legend(loc="lower right")
plt.savefig('history.png')