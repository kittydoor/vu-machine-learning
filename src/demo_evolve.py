from evolve import Evolver
from deep_nn import DeepNN

def f(x):
    return x*x

def main():
    ev = Evolver(nn=DeepNN, training_data=f, test_date=f, population=10, keep=.5)
    gen = ev.first_generation()
    
    # do stuff with first gen, like testing etc

    # ------- case 1 ------- 

    # 5 generations
    for i in range(5):
        old_gen = gen
        gen = ev.next_generation(old_gen)

        # do stuff with the generation

    # ------- case 1 end ------- 

    # ------- case 2 ------- 

    # generate and only return the 20th gen
    models = ev.generate_gen_num(20)
    # do stuff with the generation

    # ------- case 2 end ------- 

    # plot stuff

if __name__ == "__main__":
    main()
