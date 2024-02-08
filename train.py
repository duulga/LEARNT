"""
 <Assembly, IR> Mapping unit
 Author: Duulga Baasantogtokh
 Date: 24/01/2024

 usage: change instance_amount then execute

 """

from OptimalMapper import output_emitter
from Tokenizer import tokenizer, vectorizer, normalizer
import pickle
import random
import tensorflow as tf
import posenc
import matplotlib.pyplot as plt
from transformer import transformer, masked_accuracy, masked_loss, CustomSchedule

pairs = normalizer(output_emitter(150))
with open("pairs.pickle", "wb") as fp:
    pickle.dump(pairs,fp)

#Load normalized basic block pairs
with open("pairs.pickle", "rb") as fp:
    bb_pairs = pickle.load(fp)

# train-val-test split of randomized basic block pairs
random.shuffle(bb_pairs)
n_val = int(0.15*len(bb_pairs))
n_train = len(bb_pairs) - 2*n_val
train_pairs = bb_pairs[:n_train]
val_pairs = bb_pairs[n_train: n_train+n_val]
test_pairs = bb_pairs[n_train+n_val:]


def format_dataset(asm, ir):

    source = {"encoder_inputs": asm, "decoder_inputs": ir[:, :-1]}
    target = ir[:, 2:]
    return (source, target)


def format_dataset(asm, ir):

    source = {"encoder_inputs": asm, "decoder_inputs": ir[:, :-1]}
    #print(f'DEBUGDEBUGDEBUG {source["decoder_inputs"]}')
    target = ir[:, 2:]
    return (source, target)

def make_dataset(pairs, batch_size=64):
    # Create Tensorflow Dataset for the basic block pairs
    # unpack the basic blocks
    asm_tokens, ir_tokens, asm_iv_dict, ir_iv_dict, tokenized_pairs = tokenizer(pairs)
    asm_bbs = [pair[0] for pair in tokenized_pairs]
    ir_bbs = [pair[1] for pair in tokenized_pairs]
    _, _, asm_vectorized = vectorizer(asm_tokens, asm_bbs)
    _, _, ir_vectorized = vectorizer(ir_tokens, ir_bbs)
    #vectorized = format_dataset(asm_tokens, ir_tokens, asm_bbs, ir_bbs)
    #Create tensors
    #print(f'DEBUGDEBUG {asm_vectorized}')
    #print(f'DEBUGDEBUG {ir_vectorized}')
    dataset = tf.data.Dataset.from_tensor_slices((asm_vectorized, ir_vectorized))
    return dataset.shuffle(2048).batch(batch_size).map(format_dataset).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

#test the dataset
# for inputs, targets in train_ds.take(1):
#     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
#    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
#     print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
#     print(f"targets.shape: {targets.shape}")
#     print(f"targets[0]: {targets[0]}")

#test the dataset
# for inputs, targets in train_ds.take(1):
#     print(inputs["encoder_inputs"])
#     print(f"its shape: {inputs['encoder_inputs'].shape}")
#     vocab_size_asm = 10000
#     seq_length = inputs['encoder_inputs'].shape[1]
#     embed_asm = posenc.PositionalEmbedding(seq_length, vocab_size_asm, embed_dim = 512)
#     asm_emb = embed_asm(inputs["encoder_inputs"])
#     print(asm_emb.shape)
#     print(asm_emb)

vocab_size_en = 10000
vocab_size_fr = 20000
seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 128
ff_dim = 512
dropout = 0.1
vocab_size_asm = 10000
vocab_size_fr = 20000

model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                    vocab_size_en, vocab_size_fr, dropout)
lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
epochs = 20
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

#Save the trained model
model.save("asm-llvm-transformer.h5")

# Plot the loss and accuracy history
fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True)
fig.suptitle('Traininig history')
x = list(range(1, epochs+1))
axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
axs[0].set_ylabel("Loss")
axs[0].legend(loc="upper right")
axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="acc")
axs[1].plot(x, history.history["val_masked_accuracy"], alpha=0.5, label="val_acc")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("epoch")
axs[1].legend(loc="lower right")
plt.show()
