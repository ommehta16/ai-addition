if __name__ == "__main__": print("loading...")

import numpy as np
import os, sys, random, json, datetime

N = 500 # number of agents
T = 500 # number of testcases for each agent
G = 100 # number of generations
L = 4 # number of layers
SURVIVE_RATIO = 20 # 1/SURVIVE_RATIO of the population survives after each generation

assert SURVIVE_RATIO % L == 0 # I'm too lazy to implement variated reproduction

training_storage = "training"

structure = [2,3,4,1]

brains:list[list[np.ndarray]] = [ [[]] for _ in range(N) ]

for b in range(N):
    brain:list[np.ndarray] = [ [] for _ in range(L-1) ]
    for i in range(L-1):
        brain[i] = np.random.random((structure[i],structure[i+1])).astype(float)
    
    brains[b] = brain

def solve(brain:list[np.ndarray],*inp:float) -> list:
    layers:list[list[float]] = [
        [0.0 for _ in range(structure[i])] for i in range(L)
    ]


    assert len(layers[0]) == len(inp)
    layers[0] = list(inp)
    for l in range(0,L-1):
        # print(f"{l} -> {brain[l]}")
        for i in range(structure[l+1]):
            for j in range(structure[l]):
                
                layers[l+1][i] += brain[l][j,i] * layers[l][j]

    return layers[-1]

TCS = [[random.randint(0,1000),random.randint(0,1000)] for _ in range(T)]

prev_best = []

def evolve(gen:int):
    global brains, prev_best

    error_marked = []
    m_err = 1e18
    for i in range(N):
        err = 0
        for tc in range(T):
            a, b = TCS[tc]
            ans = a+b

            res = solve(brains[i],a,b)[0]
            err += (res-ans)*(res-ans)
        
        m_err = min(m_err, err)
        error_marked.append( [err, brains[i]] )

    error_marked.sort()
    # print(error_marked)
    print(m_err/T) # STDEV-like variation from "correctness"
    
    survived:list[np.ndarray] = [error_marked[i][1] for i in range(int(N/SURVIVE_RATIO))]

    mutated = []

    for parent in survived:
        for _ in range(SURVIVE_RATIO - 1):
            child = parent.copy()

            for l in range(len(child)):
                STRENGTH = max(1,5/gen) * 0.1 # how random it is -- small kickstart (doesn't do much though :/ )
                child[l] = child[l] + STRENGTH*np.random.random_sample(child[l].shape) - STRENGTH/2

                child[l] = np.clip(child[l],-2.0,2.0)
            mutated.append(child)
    survived += mutated

    with open(f"{training_storage}/gen{gen}.json",'w') as f:
        tmp = [
            [
                brains[b][l].tolist() for l in range(len(brains[b]))
            ] for b in range(N)
        ]
        json.dump(tmp,f)
    brains = survived

def run():
    to_run_path = sys.argv[2]

    if not os.path.exists(to_run_path):
        print("File does not exist")
        sys.exit(1)


    with open(to_run_path) as f: brains = json.load(f)

    brain_list = brains[0]
    assert L-1 == len(brain_list)

    brain = [[] for _ in range(len(brain_list))]

    for i in range(L-1): brain[i] = np.asarray(brain_list[i])

    while True:
        inp = input("Enter two numbers: ")
        if len(inp.split()) != 2:
            print("bro those aren't numbers what :sob:")
            continue
        try:
            a, b = map(float,inp.split())
        except:
            print("ur input stinky")
            continue
        
        print(a+b)
        print(float(solve(brain,a,b)[0]))

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("""
              Please enter atleast 1 arg:
              -run [filename]        to test inputs for the file
              -train        to train
              """)
        sys.exit(1)
    if sys.argv[1] == "-run":
        run()
    elif sys.argv[1] == "-train":
        print("started training...")
        training_storage = datetime.datetime.now().strftime("training-%m-%d-%I%p")
        if not os.path.exists(training_storage): os.mkdir(training_storage)
        for g in range(1,G+1): evolve(g)
        sys.exit()