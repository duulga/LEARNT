from OptimalMapper import output_emitter

corpus = './corpus.txt'

with open(corpus, 'r') as file:
    elements = []
    for line in file: 
        elements.append(line.split())
    
    for fuck in elements:
        #print(fuck)
        pass

pairs = output_emitter(10)
for items in pairs:
    print(items[0])
    print(items[1])
