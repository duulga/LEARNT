
file_path='/home/yoonyoseob/LEARNT/Joseph_operation/ghidra_scripts/GhidraAssembly/1/gernerated_asm1.txt'
with open(file_path, 'r') as file:
    origin_codes = file.read()

#정보 저장 클래스 
class InsParser:
    def __init__(self,function_names,bb_addrs,instructions):
        self.functions_names = function_names
        self.bb_addrs = bb_addrs
        self.instructions = instructions


#mov rbp,rsp 같은것들 normalization 
def normalizer(data):
    assembly= data.replace('[','')
    assembly= assembly.replace(']','')
    assembly = assembly.replace('\n\n',' [NC] ')
    assembly=assembly.replace('\n',' ')
    assembly = assembly.replace(',',' ')

    return assembly

# 함수 이름 추출 함수
def setFuncName(data):
    assembly=data.split()
    Function_names=[]
    # Function:을 target 해서 그 다음 인자(함수이름)를 리스트화
    targeting_Function = 'Function:'
    # Function: 이라는 내용을 찾을시 다음번 단어를 리스트에 저장
    for i, word in enumerate(assembly):
        if targeting_Function == word and i + 1 < len(assembly):
            target=assembly[i+1]
            Function_names.append(target)
    return Function_names

def setBBaddrs(data):
    assembly=data.split()
    # print(assembly)
    BBaddr_list=[]
    targeted_BBaddr_list=[]
    # Start:을 target 해서 그 다음 인자(메모리)를 리스트화
    targeting_start = 'Start:'
    targeting_end='[NC]'

    # Start: 이라는 내용을 찾을시 다음번 단어를 리스트에 저장
    for i, word in enumerate(assembly):
        #Start:를 찾으면 메모리 주소 저장
        if targeting_start == word and i + 1 < len(assembly):
            target=assembly[i+1]
            targeted_BBaddr_list.append(target)
        # [NC]를 찾고 다음이 Function인 경우 하나의 함수에 대한 BB 리스트 끝내기
        if word == targeting_end and 'Function:' == assembly[i+1]:
            BBaddr_list.append(targeted_BBaddr_list)
            targeted_BBaddr_list=[]
    return BBaddr_list

def setInstructions(data):
    assembly=data.split()
    #모든 함수 내재
    All_Instructions_list=[]
    #for문 한번 돌았을때 생기는 리스트, 즉, 한개의 BB별 리스트 
    Instructions_list = []
    per_Instructions_list = []
    targeting_start='Start:'
    targeting_end='[NC]'
    targeting_new_func='Function:'
    flag=0
    i=0
    while i < len(assembly):
        # Start:을 target 해서 그 다음, 다음 인자(basic block 첫 인자)를 리스트화
        word = assembly[i]
        if targeting_start == word and i + 1 < len(assembly):
            flag=1
            i += 2
            target=assembly[i]
            Instructions_list.append(target)
        #첫번째 이후로 flag가 계속 유지시 list에 word를 계속 추가
        if flag == 1 and word != 'Start:' and word != '[NC]':
            Instructions_list.append(word)
        #[NC]를 발견했을때, flag를 0으로 하고 지금까지 만든 리스트를 All 리스트에 추가
        if targeting_end == word:
            flag=0
            per_Instructions_list.append(Instructions_list)
            Instructions_list=[]
        if i+1 < len(assembly) and targeting_new_func == assembly[i+1]:
            All_Instructions_list.append(per_Instructions_list)
            per_Instructions_list=[]
        i += 1
    return All_Instructions_list

normalized_code=normalizer(origin_codes)
functionNames=setFuncName(normalized_code)
bbAddrs=setBBaddrs(normalized_code)
bbInstructions= setInstructions(normalized_code)
offset1 = len(functionNames) - len(bbAddrs)
offset2 = len(functionNames) - len(bbInstructions)

#갯수 차이 보충
for i in range(offset1):
    bbAddrs.append([])


for i in range(offset2):
    bbInstructions.append([])


instruction_list = [
    InsParser( function_names=functionName, bb_addrs=bbAddr, instructions=instruction) 
    for functionName, bbAddr, instruction in zip(functionNames, bbAddrs, bbInstructions)
]

print(instruction_list[0].functions_names)
print(instruction_list[0].bb_addrs)
print(instruction_list[0].instructions)