"""  
    Measurements, Checks
    To improve the model
"""
import pickle
# import matplotlib.pyplot as plt
import random
from Tokenizer import tokenizer, vectorizer, normalizer, orig_tokenizer
from OptimalMapper import output_emitter

def check_length_distribution():
    # Load normalized basic block pairs
    with open("temp.pickle", "rb") as fp:
        bb_pairs = pickle.load(fp)

    asm_size_spec = {i: 0 for i in range(1,151)}
    ir_size_spec = {i: 0 for i in range(1,151)}

    # Check the length of first element

    for entry in bb_pairs:
        length = len(entry[0].split())
        if(length < 151):
            ir_size_spec[length] = ir_size_spec[length] + 1
        length = len(entry[1].split())
        if(length < 151):
            asm_size_spec[length] = asm_size_spec[length] + 1

    print("IR size spectrum")
    print(ir_size_spec)
    print("ASM size spectrum")
    print(asm_size_spec)


    # Draw histograms of the values side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)

    # Histogram for the first dictionary
    axes[0].bar(asm_size_spec.keys(), asm_size_spec.values(), color='skyblue')
    axes[0].set_title('Assembly bbs length distribution')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 151)

    # Histogram for the second dictionary
    axes[1].bar(ir_size_spec.keys(), ir_size_spec.values(), color='lightgreen')
    axes[1].set_title('IR bbs length disribution')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 151)

    plt.tight_layout()

    file_path = './length_distribution.png'

    plt.savefig(file_path)
    plt.close()

def check_dataset():
    #Load normalized basic block pairs
    with open("temp.pickle", "rb") as fp:
        bb_pairs = pickle.load(fp)
    fp.close()

    random.shuffle(bb_pairs)
    asm_tokens, ir_tokens, tokenized_pairs = tokenizer(bb_pairs)
    with open("asm_tokens.txt", "w") as fp:
        for token in asm_tokens:
            fp.write(f'{token}\n')
    with open("ir_tokens.txt", "w") as fp:
        for token in ir_tokens:
            fp.write(f'{token}\n')
    with open("tokenized_pairs.txt", "w") as fp:
        for tokenized_asm, tokenized_ir in tokenized_pairs:
            fp.write(f'{tokenized_asm}\n')
            fp.write(f'{tokenized_ir}\n')

def get_vectorized_example():
    #Load normalized basic block pairs
    with open("temp.pickle", "rb") as fp:
        bb_pairs = pickle.load(fp)
    fp.close()

    asm_tokens, ir_tokens, tokenized_pairs = tokenizer(bb_pairs[0:1])  
    asm_bbs = [pair[0] for pair in tokenized_pairs]
    ir_bbs = [pair[1] for pair in tokenized_pairs]

    asm_dic, _, asm_vectorized = vectorizer(asm_tokens, asm_bbs, 80)
    ir_dic, _, ir_vectorized = vectorizer(ir_tokens, ir_bbs, 80)

    print(asm_bbs)
    print(ir_bbs)
    asm_temp = ''
    for item in asm_vectorized[0]:
        asm_temp = asm_temp + str(item) + ' '
    print(asm_temp)
    ir_temp = ''
    for item in ir_vectorized[0]:
        ir_temp = ir_temp + str(item) + ' '
    print(ir_temp)

    print(len(asm_vectorized[0]))
    print(len(ir_vectorized[0]))

def count_pairs():
    #Load normalized basic block pairs
    with open("temp.pickle", "rb") as fp:
        bb_pairs = pickle.load(fp)
    fp.close()

    print(f' Total number of pairs: {len(bb_pairs)}')

def check_function_placeholder(corpus_size):
    pairs = normalizer(output_emitter(corpus_size))

    asm_tokens, ir_tokens, new_tokenized_pairs = tokenizer(pairs)            
    with open("asm_tokens.txt", "w") as ff:
        for apair in asm_tokens:
            ff.write(f"{apair}\n")

def generate_dictionaries(corpus_size):
    pairs = normalizer(output_emitter(corpus_size))
    asm_tokens, ir_tokens, tokenized_pairs = orig_tokenizer(pairs)
    # asm_bbs = [pair[0] for pair in tokenized_pairs]
    ir_bbs = [pair[1] for pair in tokenized_pairs]
    # ASM_DIC, _, asm_vectorized = vectorizer(asm_tokens, asm_bbs, 80)
    IR_DIC, _, ir_vectorized = vectorizer(ir_tokens, ir_bbs, 80)
    # ASM_VOCAB_SIZE = len(ASM_DIC)
    # IR_VOCAB_SIZE = len(IR_DIC)

    # with open("asm_dic.pickle", "wb") as fp:
    #     pickle.dump(ASM_DIC,fp)

    for key in IR_DIC:
        print(f'{key}: {IR_DIC[key]}')

    with open("data/ir_dic.pickle", "wb") as fp:
        pickle.dump(IR_DIC,fp)

def create_modified_asm_corpus(corpus_size):
    # Create modified version of corpus for PalmTree
    pairs = output_emitter(corpus_size)
    asm_bbs = [pair[1] for pair in pairs]
    new_bbs = []
    for bb in asm_bbs:
        #if(',' in bb):
        bb = bb.replace(',', '')
        #if('[' in bb):
        bb = bb.replace('[', '[ ')
        #if(']' in bb):
        bb = bb.replace(']', ' ]')
        #if('_' in bb):
        bb = bb.replace('-', ' - ')
        temp = bb.split('\n')
        new_bbs.append(temp)
    for bbs in new_bbs:
        for instructions in bbs:
            print(instructions)
    with open(f"data/asm-{corpus_size}.pickle", "wb") as fi:
        pickle.dump(new_bbs, fi)

def create_ir_corpus(corpus_size):
    pairs = normalizer(output_emitter(corpus_size))
    ir_bbs = [pair[0] for pair in pairs]
    with open(f"data/ir-{corpus_size}.pickle", "wb") as fp:
        pickle.dump(ir_bbs,fp)
    for i in ir_bbs:
        print(f'za bas arai: {i}')
    print("Done!")

def check_pickle(pickle_path):
    with open(pickle_path, 'rb') as dfeq:
        data = pickle.load(dfeq)
    for x in data:
        print(f'{x}: {data[x]}')

create_ir_corpus(5000)
create_modified_asm_corpus(5000)
# generate_dictionaries(50000)
# check_pickle('data/ir_dic.pickle')