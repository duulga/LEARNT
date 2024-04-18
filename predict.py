import random
import pickle

import numpy as np
import tensorflow as tf

from Tokenizer import imm_val_tokenization, vectorizer, single_tokenizer_single_bb, vectorizer_with_dic_single_bb
from posenc import PositionalEmbedding
from trainprep import CustomSchedule, masked_loss, masked_accuracy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MAX_BB_LENGTH = 80

# Load normalized basic block pairs
with open("temp.pickle", "rb") as fp:
    bb_pairs = pickle.load(fp)

# train-val-test split of randomized basic block pairs
random.shuffle(bb_pairs)
n_val = int(0.15*len(bb_pairs))
n_train = len(bb_pairs) - 2*n_val
test_pairs = bb_pairs[n_train+n_val:]

# Vectorize the whole dataset!
# print("Getting the size of vocabs")
# asm_tokens, ir_tokens, tokenized_pairs = tokenizer(bb_pairs)
# asm_bbs = [pair[0] for pair in tokenized_pairs]
# ir_bbs = [pair[1] for pair in tokenized_pairs]
# ASM_DIC, _, asm_vectorized = vectorizer(asm_tokens, asm_bbs, MAX_BB_LENGTH)
# IR_DIC, _, ir_vectorized = vectorizer(ir_tokens, ir_bbs, MAX_BB_LENGTH + 1)

# with open("asm_dic.pickle", "wb") as fp:
#     pickle.dump(ASM_DIC,fp)

# with open("ir_dic.pickle", "wb") as fp:
#     pickle.dump(IR_DIC,fp)

# Fetching the global vocabulary!
with open("asm_dic.pickle", "rb") as fp:
    ASM_DIC = pickle.load(fp)

with open("ir_dic.pickle", "rb") as fp:
    IR_DIC = pickle.load(fp)

# Checking if there is our start sentinel in vocabulary
# print(f'~~~IR_DIC["[start]"]: {IR_DIC["[start]"]} ')

custom_objects = {"PositionalEmbedding": PositionalEmbedding,
                  "CustomSchedule": CustomSchedule,
                  "masked_loss": masked_loss,
                  "masked_accuracy": masked_accuracy
                  }

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model("asm-llvm-transformer_checkpoint.h5")

#training parameters used
seq_len = MAX_BB_LENGTH
asm_vocab_size = 1002
ir_vocab_size = 6364

def translate(basic_block):
    """Create the translated basic block"""
    enc_tokens = imm_val_tokenization(basic_block)
    asm_vectors, _ = vectorizer_with_dic_single_bb(ASM_DIC, enc_tokens, seq_len)

    lookup = list(IR_DIC)
    # print(IR_DIC.keys())
    # print(IR_DIC.values())
    # print(lookup)
    start_sentinel, end_sentinel = "[start]", "[end]"
    vector_output = [start_sentinel]
    output_bb = [start_sentinel]
    #generate the translated bb token by token
    for i in range(seq_len):
        tokenized = imm_val_tokenization(" ".join(output_bb))
        ir_vectors, _ = vectorizer_with_dic_single_bb(IR_DIC, tokenized, seq_len+1)
        # print(f'FOR DEBUG: Going {i}th')
        assert ir_vectors.shape == (1, seq_len+1)
        dec_tokens = ir_vectors[:, :-1]
        assert dec_tokens.shape == (1, seq_len)
        # print(f'FOR DEBUG~ asm_vectors: {asm_vectors}th')
        # print(f'FOR DEBUG~ dec_tokens: {dec_tokens}th')
        pred = model([asm_vectors, dec_tokens])
        assert pred.shape == (1, seq_len, ir_vocab_size)
        the_word = np.argmax(pred[0, i, :])
        vector_output.append(str(the_word))
        word = lookup[the_word-1]
        output_bb.append(word)
        if word == end_sentinel:
            break
    return output_bb, vector_output, enc_tokens

test_count = 20
for n in range(test_count):

    input_ir_bb, input_asm_bb = random.choice(test_pairs)
    temp_ir, temp_asm = input_ir_bb.split()[:80], input_asm_bb.split()[:80]
    ir_bb, asm_bb = " ".join(temp_ir), " ".join(temp_asm)

    # print(f'FOR DEBUG~ asm_bb: {asm_bb}')
    translated, vector_translated, enc_tokens = translate(asm_bb)
    tokenized = imm_val_tokenization(ir_bb)
    print(f"Test {n+1}:")
    print(f"{enc_tokens}")
    print(f"== {tokenized}")
    print(f":) {' '.join(vector_translated)}")
    print(f"-> {' '.join(translated)}")
    print()