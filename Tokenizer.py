"""
 <Assembly, IR> Mapping unit
 Author: Duulga Baasantogtokh
 Date: 13/12/2023
 """
import numpy 

def normalizer(bb_pairs):
    normalized = []
    for ir,assembly in bb_pairs:
        assembly = assembly.replace('\n', ' [NC] ')
        assembly = assembly.replace(',', '')
        ir = ir.replace(',', '')
        ir = ir.replace('\n', ' [NC] ')
        ir = ir.replace('(', ' ')
        ir = ir.replace(')', '')
        pair = (ir, assembly)
        normalized.append(pair)
        # print(f"DEBUGDEBUG {ir}")
        # print(f"DEBUGDEBUG {assembly}")
    #print(normalized)
    return normalized

def imm_val_tokenization(bb):
    """ 
        Tokenize immediate values 
        of the input basic block
    """
    immediate_values = {}
    
    tokens = bb.split()

    new_bb = ""
    iv_count = 0

    for token in tokens:
        try: 
            temp = int(token)
            token = str(temp)
            if(token in immediate_values.values()):
                for alr_tokens in immediate_values:
                    if(immediate_values[alr_tokens] == token):
                        tokens[tokens.index(token)] = alr_tokens
                    else: pass
                pass
            else:
                new_token = "iv" + str(iv_count)
                immediate_values[new_token] = token
                iv_count += 1
                tokens[tokens.index(token)] = new_token

        except ValueError:
            pass
    
    return " ".join(tokens)

def tokenizer(bb_pairs):
    tokenized_pairs = []
    assembly_tokens, ir_tokens = set(), set()
    # FOP: From one program
    assemblies_fop, irs_fop = "", ""
    for ir, assembly in bb_pairs:
        # print(f"###DEBUG Assembly in pair: {assembly}")
        # print(f"###DEBUG IR in pair: {ir}")
        if(ir.split()[0] == '[PROGRAM_END]'):
            assembly_tok, ir_tok = imm_val_tokenization(assemblies_fop).split(), imm_val_tokenization(irs_fop).split()
            # print(f"###DEBUG Assembly_tok: {assembly_tok}")
            # print(f"###DEBUG ir_tok: {ir_tok}")
            # FOP: From one program
            assemblies_fop, irs_fop = "", ""
            assembly_tokens.update(assembly_tok)
            ir_tokens.update(ir_tok)
            # b2bb: back to basic block
            b2bb_asm = back_to_bb(assembly_tok)
            b2bb_ir = back_to_bb(ir_tok)
            for i in range(0, len(b2bb_asm)):
                pairs = []
                pairs.append(b2bb_asm[i])
                pairs.append(b2bb_ir[i])
                tokenized_pairs.append(pairs)

        else:
            # print(f"DEBUG assemblies_fop: {assemblies_fop}")
            assemblies_fop = assemblies_fop + " " + assembly + '[bb]'
            irs_fop = irs_fop + " " + ir + " " + '[bb]'
        #print(f"###DEBUG Assembly after split {assembly_tok}")
        #print(f"###DEBUG IR after split {ir_tok}")
    return sorted(assembly_tokens), sorted(ir_tokens), tokenized_pairs

def back_to_bb(abstracted_bbs):
    pairs = []
    # Back to basic block
    b2bb = []
    #empty list
    empty = []
    for token in abstracted_bbs:
        if(token == '[bb]'):
            pairs.append(" ".join(b2bb))
            b2bb = empty
        else:
            b2bb.append(token)
    return pairs
        
def vectorizer_with_dic(dic, bbs, length):
    new_bbs = []
    
    for bb in bbs:
        new_bb = []
        for token in bb.split():
            new_bb.append(dic[token])
        new_bbs.append(new_bb)

    max = length
    for i in range(len(new_bbs)):
        size = len(new_bbs[i]) 
        if(size > max):
            new_bbs[i] = new_bbs[i][:max]
        else:
            padding = max - size
            for _ in range(padding):
                new_bbs[i].append(0)
    
    arr = numpy.array(new_bbs, dtype="object")
    #print(f"DEBUGDEBUG {arr}")
    return arr, new_bbs

def vectorizer_with_dic_single_bb(dic, bb, length):

    new_bb = []
    for token in bb.split():
        new_bb.append(dic[token])

    max = length
    size = len(new_bb)
    if(size > max):
        new_bb = new_bb[:max]
    else:
        padding = max - size
        for _ in range(padding):
            new_bb.append(0)
    
    arr = numpy.array(new_bb, dtype="object")
    arr = numpy.reshape(arr, (1, length))
    arr = arr.astype(numpy.float32)
    #print(f"DEBUGDEBUG {arr}")
    return arr, new_bb

def vectorizer(tokens, bbs, length):
    vector_dict = {}
    vector = 1
    for token in tokens:
        vector_dict[token] = vector
        vector += 1

    count = 0
    new_bbs = []
    for bb in bbs:
        new_bb = []
        for token in bb.split():
            new_bb.append(vector_dict[token])
        #bbs[count] = new_bb
        new_bbs.append(new_bb)
        count += 1
    #Finding longest basic block
    #max = len(new_bbs[0])
    # for i in new_bbs:
    #     print(f"#DEBUG {len(i)}" )
    #     length = len(i)
    #     if(max < length):
    #         max = length
    max = length
    #print(f"Longest: {max}")

    #Make the array homogenous
    for i in range(len(new_bbs)):
        size = len(new_bbs[i])
        if(size > max):
            new_bbs[i] = new_bbs[i][:max]
        else:
            padding = max - size
            for _ in range(padding):
                new_bbs[i].append(0)
    
    arr = numpy.array(new_bbs, dtype="object")
    #print(f"DEBUGDEBUG {arr}")
    return vector_dict, arr, new_bbs

def single_tokenizer(bbs):
    tokenized_bbs = []
    tokens = set()
    immediate_values = {}
    iv_count = 0

    bb_tok = bbs.split()

    for token in bb_tok:
        try:
            temp = int(token)
            token = str(temp)
            if(token in immediate_values.values()):
                index = 0
                for alr_tokens in immediate_values:
                    if(immediate_values[alr_tokens] == token):
                        #print(f"DEBUGDEBUG token: {token} alr_token: {alr_tokens}")
                        bb_tok[bb_tok.index(token)] = alr_tokens
                    else: pass
            else:
                new_token = "iv" + str(iv_count)
                immediate_values[new_token] = token
                #print(f"DEBUGDEBUG token: {token} replace_token: {new_token}")
                iv_count += 1
                bb_tok[bb_tok.index(token)] = new_token

        except ValueError:
            pass
    
        tokens.update(bb_tok)
        tokenized_bbs.append(" ".join(bb_tok))
    
    return tokenized_bbs

def single_tokenizer_single_bb(bb):
    tokens = set()
    bb_tok = imm_val_tokenization(bb).split()
    tokens.update(bb_tok)
    
    return " ".join(bb_tok)

    
    
    

    
