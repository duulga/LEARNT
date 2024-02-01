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
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
#    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
#     print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
#     print(f"targets.shape: {targets.shape}")
#     print(f"targets[0]: {targets[0]}")




#test the dataset
for inputs, targets in train_ds.take(1):
    print(inputs["encoder_inputs"])
    print(f"its shape: {inputs['encoder_inputs'].shape}")
    vocab_size_asm = 10000
    seq_length = inputs['encoder_inputs'].shape[1]
    embed_asm = posenc.PositionalEmbedding(seq_length, vocab_size_asm, embed_dim = 512)
    asm_emb = embed_asm(inputs["encoder_inputs"])
    print(asm_emb.shape)
    print(asm_emb)


