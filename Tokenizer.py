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

def tokenizer(bb_pairs):
    tokenized_pairs = []
    assembly_tokens, ir_tokens = set(), set()
    immediate_values_asm = {}
    immediate_values_ir = {}
    iv_count_asm = 0
    iv_count_ir = 0
    for ir, assembly in bb_pairs:
        #print(f"###DEBUG Assembly in pair: {assembly}")
        #print(f"###DEBUG IR in pair: {ir}")=
        assembly_tok, ir_tok = assembly.split(), ir.split()
        #print(f"###DEBUG Assembly after split {assembly_tok}")
        #print(f"###DEBUG IR after split {ir_tok}")

        for token in assembly_tok:
            try:
                temp = int(token)
                token = str(temp)
                if(token in immediate_values_asm.values()):
                    index = 0
                    for alr_tokens in immediate_values_asm:
                        if(immediate_values_asm[alr_tokens] == token):
                            #print(f"DEBUGDEBUG token: {token} alr_token: {alr_tokens}")
                            assembly_tok[assembly_tok.index(token)] = alr_tokens
                        else: pass
                else:
                    new_token = "iv" + str(iv_count_asm)
                    immediate_values_asm[new_token] = token
                    #print(f"DEBUGDEBUG token: {token} replace_token: {new_token}")
                    iv_count_asm += 1
                    assembly_tok[assembly_tok.index(token)] = new_token

            except ValueError:
                pass
        
        for token in ir_tok:
            try:
                temp = int(token)
                token = str(temp)

                if(token in immediate_values_ir.values()):
                    for alr_tokens in immediate_values_ir:
                        if(immediate_values_ir[alr_tokens] == token):
                            #print(f"DEBUG token : {token} alr_token: {alr_tokens}")
                            ir_tok[ir_tok.index(token)] = alr_tokens
                        else: pass
                else:
                    new_token = "iv" + str(iv_count_ir)
                    immediate_values_ir[new_token] = token
                    #print(f"DEBUGDEBUG token: {token} replace_token: {new_token}")
                    iv_count_ir += 1
                    ir_tok[ir_tok.index(token)] = new_token

            except ValueError:
                pass

        assembly_tokens.update(assembly_tok)
        ir_tokens.update(ir_tok)
        pairs = []
        pairs.append(" ".join(assembly_tok))
        pairs.append(" ".join(ir_tok))
        tokenized_pairs.append(pairs)
    return sorted(assembly_tokens), sorted(ir_tokens),immediate_values_asm, immediate_values_ir, tokenized_pairs

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


    

    
