from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from BPSO import *
from model import *
from utils import *

def fitness_without_optimization(train, test):
    model = create_model_single()
    model.fit(train, steps_per_epoch=128, epochs=10, verbose=1)
    
    class_acc = model.fit(test, steps_per_epoch=128, verbose=1)
    return (class_acc.history['accuracy'], class_acc.history['precision'], class_acc.history['recall'])

def pso_calculate(f_count, X, x_t):
    y_actual = []
    y_predict = []
    fitness_best_g = (-1, -1, -1)
    pos_fitness_g = []
    swarm = []
    no_population = 400
    
    for i in range(no_population):
        swarm.append(BPSO(f_count, X, x_t))
    
    #optimize 
    index = 5
    for index in tqdm(range(index)):
        for pos in tqdm(range(no_population)):
            swarm[pos].evaluate_fitness()
            
            #check current particle is the global best 
            if swarm[pos].fitness_check(swarm[pos].fitness, fitness_best_g): #swarm[pos].fitness > fitness_best_g or fitness_best_g == -1:
                pos_fitness_g = list(swarm[pos].position)
                fitness_best_g = (swarm[pos].fitness)
                score = swarm[pos].score_best
                  
        for pos in range(no_population):
            swarm[pos].update_velocity(pos_fitness_g)
            swarm[pos].update_position()
    
    print('\n Final Solution:')
    print(pos_fitness_g)
    print(fitness_best_g)
    print(score)