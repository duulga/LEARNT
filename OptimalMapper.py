"""
 <Assembly, IR> Mapping unit
 Author: Duulga Baasantogtokh
 Date: 13/12/2023

 TODO list
 1. Found out that there exist a basic block in IR which does not exist in assembly 
    - Change assembly BB names to llc generated annotations
    - Then, compare extracted assembly and IR BB dictionaries 
    - Eliminate IR BBs without matching pairs

 """

#AMOUNT OF RANDOM SAMPLES GENERATED
INSTANCE_AMOUNT = 1000

def parse_functions_from_assembly(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    functions = {}
    current_func = None
    func_started = False

    for line in lines:
        # Check for function start (assuming a specific comment format)
        if "# -- Begin function" in line:
            func_started = True
            current_func = line.strip().split(' ')[-1]  # Get function name
            functions[current_func] = []

        elif func_started:
            # Add lines to the current function
            functions[current_func].append(line)
            
            # Check for function end (assuming a specific comment format)
            if "# -- End function" in line:
                # Cutting function epilogue part for convenience
                functions[current_func] = functions[current_func][:-3]
                func_started = False
                current_func = None
                
    return functions

def parse_functions_from_llvm_ir(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    functions = {}
    current_func = None
    func_body = []

    for line in lines:
        if line.strip().startswith('define'):
            if current_func is not None:
                # Save the previous function
                functions[current_func] = func_body
                func_body = []

            # Start a new function
            # Find the position of "@" and extract the substring after it
            at_pos = line.find("@")
            at_pos_1 = line.find("(")
            current_func = line[at_pos + 1:at_pos_1].strip() # Get function name

            func_body.append(line)

        elif current_func is not None:
            # Add lines to the current function's body
            func_body.append(line)

            if line.strip() == '}':
                # Function end, save it and reset
                functions[current_func] = func_body
                current_func = None
                func_body = []

    # Handle the last function if the file doesn't end with a newline
    if current_func is not None:
        functions[current_func] = func_body

    return functions

def extract_BBs_assembly(function):
    isBody = False
    extracted_BBs = {}
    bb_body = ""
    bb_name = ""
    
    for line in function:
        if(line[0] == "\t"):
            #if(isBody):
            if(line[1] != '.'):
            #print(bb_body)
                bb_body += line.strip("\t")
        else:
            extracted_BBs[bb_name] = bb_body
            bb_body = ""

        #Locate the start of the basic block
        at_pos = line.find("#")
        if(at_pos != -1 and line[at_pos + 2] == "%"):
            elements = line.split()
            isBody = True
            if(at_pos == 0):
                bb_name = elements[3][1:]
            else:
                bb_name = elements[2][1:]

    extracted_BBs.pop("")
    return extracted_BBs

    """ for bb_names, bb_bodies in extracted_BBs.items():
        print(f'BASIC BLOCK: {bb_names}')
        print(f'BOOOOODYYY: {bb_bodies}')
        print('=========================') """

def extract_BBs_ir(function):
    isBody = False
    extracted_BBs = {}
    bb_body = ""
    bb_name = ""
    for line in function:
        if(line[0] == " " or line[0] == "\n"):
            if(isBody):
                at_pos = line.find("align")
                if(at_pos != -1):
                    line = line[:at_pos - 2] + "\n"
                    bb_body += line.strip(" ")
                else:
                    bb_body += line.strip(" ")
                
        else:
            bb_body = bb_body + '\n'
            extracted_BBs[bb_name] = bb_body
            bb_body = ""

        #Locate the start of the basic block
        at_pos = line.find(":")
        if(at_pos != -1):
            isBody = True
            bb_name = line.split()[0][:-1]

    extracted_BBs.pop("")
    return extracted_BBs

def output_emitter(instance_amount):
    ir_count = 1
    assembly_count = 1
    pairs = []
    for i in range(1, instance_amount+1):
        i = str(i)
        assembly_file_path = './corpus/instance' + i + '/random' + i + '.s'
        ir_file_path = './corpus/instance' + i + '/random' + i + '.ll'
        assembly_functions = parse_functions_from_assembly(assembly_file_path)
        ir_functions = parse_functions_from_llvm_ir(ir_file_path)

        ir_BBs = {}
        assembly_BBs = {}
        base_count = ir_count
        temp_count = 1
        for function_name in ir_functions:
                extracted_ir_BBs = extract_BBs_ir(ir_functions[function_name])
                extracted_assembly_BBs = extract_BBs_assembly(assembly_functions[function_name])

                for extracted_bb_name in extracted_assembly_BBs:
                    bb_name = "bb" + str(temp_count)
                    if extracted_bb_name in extracted_ir_BBs:
                        ir_BBs[bb_name] = extracted_ir_BBs[extracted_bb_name]
                        assembly_BBs[bb_name] = extracted_assembly_BBs[extracted_bb_name]
                        pair = (ir_BBs[bb_name], assembly_BBs[bb_name])
                        pairs.append(pair)
                        temp_count += 1
                    ir_count += 1
                    

        temp_count = 1 
        # print(f"###DEBUG IR dict keys {ir_BBs.keys()}")
        # print(f"###DEBUG IR dict length {len(ir_BBs)}")
        # print(f"###DEBUG Assembly dict keys {assembly_BBs.keys()}")
        # print(f"###DEBUG Assembly dict length {len(assembly_BBs)}")

        output_path = './corpus/instance' + i + '/random' + i + '.txt'

        with open(output_path, 'w') as file:
            for i in range(1, len(assembly_BBs)):
                bb_name = "bb" + str(i)
                file.write(f"$Assembly$bb{base_count}\n")
                file.write(assembly_BBs[bb_name])
                file.write(f"\n$IR$bb{base_count}\n")
                file.write(ir_BBs[bb_name])
                base_count = base_count + 1
    
    output_path = './pairs.txt'
    with open(output_path, 'w') as file:
        for items in pairs:
            file.write(items[0])
            file.write(items[1])
    
    return pairs


def corpus_emitter(instance_amount):
    output_path = './corpus.txt'
    corpus = ''
    with open(output_path , 'w') as file:
        for i in range(1, instance_amount + 1):
            i = str(i)
            file_path = './corpus/instance' + i + '/random' + i + '.txt'
            with open(file_path, 'r') as file_to_read:
                for lines in file_to_read:
                    corpus += lines
                    #file.write(lines)
        corpus = corpus.replace('\n\n\n', '\n\n')
        file.write(corpus)


output_emitter(INSTANCE_AMOUNT)
corpus_emitter(INSTANCE_AMOUNT)
