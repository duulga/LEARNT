import random
import pickle

import numpy as np
import tensorflow as tf

from Tokenizer import single_tokenizer, vectorizer_with_dic, single_tokenizer_single_bb, vectorizer_with_dic_single_bb
from posenc import PositionalEmbedding
from trainprep import CustomSchedule, masked_loss, masked_accuracy

MAX_BB_LENGTH = 150

# Load normalized basic block pairs
with open("temp.pickle", "rb") as fp:
    bb_pairs = pickle.load(fp)

# train-val-test split of randomized basic block pairs
random.shuffle(bb_pairs)
n_val = int(0.15*len(bb_pairs))
n_train = len(bb_pairs) - 2*n_val
test_pairs = bb_pairs[n_train+n_val:]

# # Vectorize the whole dataset!
# print("Getting the size of vocabs")
# asm_tokens, ir_tokens, asm_iv_dict, ir_iv_dict, tokenized_pairs = tokenizer(bb_pairs)
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
seq_len = 150
asm_vocab_size = 31528
ir_vocab_size = 34227

def translate(basic_block):
    """Create the translated basic block"""
    enc_tokens = single_tokenizer_single_bb(basic_block)
    asm_vectors, _ = vectorizer_with_dic_single_bb(ASM_DIC, enc_tokens, seq_len)

    lookup = list(IR_DIC)
    start_sentinel, end_sentinel = "[start]", "[end]"
    output_bb = [start_sentinel]
    #generate the translated bb token by token
    for i in range(seq_len):
        tokenized = single_tokenizer_single_bb(" ".join(output_bb))
        ir_vectors, _ = vectorizer_with_dic_single_bb(IR_DIC, tokenized, seq_len+1)
        # print(f'FOR DEBUG: Going {i}th')
        assert ir_vectors.shape == (1, seq_len+1)
        dec_tokens = ir_vectors[:, :-1]
        assert dec_tokens.shape == (1, seq_len)
        # print(f'FOR DEBUG~ asm_vectors: {asm_vectors}th')
        # print(f'FOR DEBUG~ dec_tokens: {dec_tokens}th')
        pred = model([asm_vectors, dec_tokens])
        assert pred.shape == (1, seq_len, ir_vocab_size)
        word = lookup[np.argmax(pred[0, i, :])]

        output_bb.append(word)
        if word == end_sentinel:
            break
    return output_bb

test_count = 20
for n in range(test_count):
    ir_bb, asm_bb = random.choice(test_pairs)
    # print(f'FOR DEBUG~ asm_bb: {asm_bb}')
    translated = translate(asm_bb)
    print(f"Test {n}:")
    print(f"{asm_bb}")
    print(f"== {ir_bb}")
    print(f"-> {' '.join(translated)}")
    print()