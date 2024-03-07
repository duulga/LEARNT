"""
 <Assembly, IR> corpus generator
 Author: Duulga Baasantogtokh
 Date: 05/12/2023

 DEPENDENCY: CSmith must be installed in the path ./csmith-install/bin/csmith 

 TODO list
 1. Generate arbitrary number of random C programs
 2. Compile all generated C programs
 3. Compile all generated C programs to executable not relocatable object files
"""

import subprocess
import argparse
import json
import time
import os

#PATHS MUST BE SET BEFORE EXECUTING!
GENERATOR_PATH = "~/csmith-install/bin/csmith"
COMPILER_PATH = "~/llvm-project/build/bin/clang "
LLC_PATH = "~/llvm-project/build/bin/llc "

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest='quantity', required=True, help="Number of random C samples to be generated")
    args = parser.parse_args()
    return args

def parse_arguments_from_json(file_path):
    with open(file_path, 'r') as file:
        data  = json.load(file)
    return data

def generate(number, config):
    #Command line options parsing
    CL_Options = parse_arguments_from_json(config)
    generator_options = " "
    for args in CL_Options.values():
        generator_options += args + " "

    output_path = "./corpus/instance"
    #Execution
    for i in range(1, number):
        output_name = output_path + str(i) + "/random" + str(i) + ".c"
        command2generate = GENERATOR_PATH + generator_options + ">" + output_name
        subprocess.run(command2generate, shell=True)
        print("Generated " + output_name)

def generate_dirs(number):
    for i in range(1, number):
        dir_name = "./corpus/instance" + str(i)
        if(os.path.exists(dir_name)):
            continue
        else:
            command2generate_dirs = "mkdir " + dir_name
            subprocess.run(command2generate_dirs, shell=True)

def compile_all(number):
    compiler_options = "-I~/csmith-install/include/ "
    #compiler_options = "-c -w -I./csmith-install/include/ "
    #~/llvm-project/build/bin/clang -I../../csmith-install/include random1.c -o random1
    for i in range(1, number):
        binary_path = "./corpus/instance" + str(i) + "/random" + str(i) + ".c "
        output_path = binary_path[:-3] + "_X64"
        command2compile = COMPILER_PATH + compiler_options + binary_path + "-o " + output_path
        #print(command2compile)
        subprocess.run(command2compile, shell=True)

def emit_llvm_all(number):
    compiler_options = "-I ~/csmith-install/include/ -S -emit-llvm -w "
    for i in range(1, number):
        target_path = "./corpus/instance" + str(i) + "/random" + str(i) + ".c "
        llvm_output_path = target_path[:-3] + ".ll"
        command2compile = COMPILER_PATH + compiler_options + target_path + "-o " + llvm_output_path
        #print(command2compile)
        subprocess.run(command2compile, shell=True)

def emit_assembly_all(number):
    compiler_options = "--x86-asm-syntax=intel "
    for i in range(1, number):
        target_path = "./corpus/instance" + str(i) + "/random" + str(i) + ".ll "
        assembly_output_path = target_path[:-4] + ".s"
        command2compile = LLC_PATH + compiler_options + target_path + "-o " + assembly_output_path
        #print(command2compile)
        subprocess.run(command2compile, shell=True)

if __name__ == "__main__":
    start = time.time()
    args = arg_parser()
    config = "./config.json"
    quantity = int(args.quantity) + 1
    generate_dirs(quantity) 
    #generate C files
    #generate(quantity, config)
    #Generate binary
    #compile_all(quantity)
    #Generate llvm IR 
    emit_llvm_all(quantity)
    #Generate target-specific assembly files
    emit_assembly_all(quantity)
    end = time.time()
    exec_time = end - start
    print("Execution time: " + str(exec_time))
