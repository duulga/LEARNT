from OptimalMapper import output_emitter
import pickle
import random

pairs = output_emitter(10)
with open("pairs.pickle", "wb") as fp:
    pickle.dump(pairs,fp)

with open("pairs.pickle", "rb") as fp:
    bb_pairs = pickle.load(fp)
    
def tokenizer(bb_pairs):
    tokenized_pairs = []
    assembly_tokens, ir_tokens = set(), set()
    immediate_values_asm = {}
    immediate_values_ir = {}
    iv_count_asm = 0
    iv_count_ir = 0
    for ir, assembly in bb_pairs:
        #print(f"###DEBUG Assembly in pair: {assembly}")
        #print(f"###DEBUG IR in pair: {ir}")
        #Some cleaning before tokenizing!
        assembly = assembly.replace('\n', ' [NC] ')
        ir = ir.replace('\n', ' [NC] ')
        ir = ir.replace('(', ' ')
        ir = ir.replace(')', '')
        assembly_tok, ir_tok = assembly.split(), ir.split()
        #print(f"###DEBUG Assembly after split {assembly_tok}")
        #print(f"###DEBUG IR after split {ir_tok}")
        # Deal with "[rbp-n]," & "[rbp-n]"
        assembly_tok = [token.replace(",", "") for token in assembly_tok]
        ir_tok = [token.replace(",", "") for token in ir_tok]

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
        pairs = (" ".join(assembly_tok), " ".join(ir_tok))
        tokenized_pairs.append(pairs)
    return sorted(assembly_tokens), sorted(ir_tokens),immediate_values_asm, immediate_values_ir, tokenized_pairs

def vectorizer(tokens):
    size = len(tokens)

# print(f"Total assembly tokens: {len(assembly_tokens)}")
# print(f"Total ir tokens: {len(ir_tokens)}")
# print(f"{len(bb_pairs)} total pairs")

asm_tokens, ir_tokens, asm_iv_dict, ir_iv_dict, tokenized_pairs = tokenizer(bb_pairs)



print(f" asm tokens: {len(asm_tokens)}")
print(f" IR tokens: {len(ir_tokens)}")
# train_pairs = bb_pairs["train"]
# val_pairs = bb_pairs["val"]
# test_pairs = bb_pairs["test"]
    
