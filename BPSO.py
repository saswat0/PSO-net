import numpy as np

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

from utils import *
from model import *

class BPSO:
    def __init__(self, f_count, X, x_t):
        
        #feature count 
        self.f_count  = f_count
        # Actual Positions  radmon prob
        self.pos_act  = []
        # Position prob > 0.5 set as 1 or 0  
        self.position = []
        # Velocity random between -1 and 1 
        self.velocity = []
        # best position 
        self.pos_best = []
        # Y actual 
        self.y_actual = []
        # Y test predicted 
        self.y_predict= []
        # best fit accuracy, Recall, Precision
        self.fit_best = (-1, -1, -1)
        # accuracy , recall, precision 
        self.fitness  = (-1, -1, -1)
        # data 
        self.X  = X
        self.x_t = x_t
        self.score_best = None 
        
        self.initialize(f_count)
    
    # initialize 
    def initialize(self, f_count):
        self.f_count = f_count
        self.initalize_position(f_count)
        self.initialize_velocity(f_count)
    
    def set_data(self,data):
        self.df = data
        print(self.df.head())
        
    #Initialize the positions > 0.5  is set as 1
    def initalize_position(self,f_count):
        self.pos_act = np.random.uniform(low=0, high=1, size=f_count).tolist()
        self.position = [1 if po > 0.5 else 0  for po in self.pos_act]
        
    def initialize_velocity(self, f_count):
        self.velocity = np.random.uniform(low=-1, high=1, size=f_count).tolist()
    
    def drop_columns(self, X):
        print(X.shape)
        print(self.position)
        for index, value in enumerate(self.position):
            if value == 0 :
                X_1 = X.drop(X.columns[index], axis = 1)
        return X_1
    
    def classification_accuracy(self, y_actual, y_hat):
        accuracy = accuracy_score(y_actual, y_hat)
        precision = precision_score(y_actual, y_hat, average="macro", labels=np.unique(y_hat))
        sensitivity = recall_score(y_actual, y_hat, average="macro", labels=np.unique(y_hat))
        return accuracy, sensitivity, precision
    
    def process_data(self):
        
        # Separate labels and features
        X = self.X
        
        x_t = self.x_t
        
        reset_keras()

        model = create_model_single()
        model.fit(X, steps_per_epoch=128, epochs=10, verbose=0)
        
        class_acc = model.fit(x_t, steps_per_epoch=128, verbose=0)
        self.score_best =  class_acc.history['accuracy'][0]
        return (class_acc.history['accuracy'][0], class_acc.history['precision'][0], class_acc.history['recall'][0])
    
    # fitness check, checks accuarcy and precision and accurarcy 
    def fitness_check(self, fitness, fit_best):
        is_fitness = False
        
        if fitness[0] > fit_best[0] or fit_best[0] == -1:
            if fitness[1] >= fit_best[1] and fitness[2] >= fit_best[2]:
                is_fitness = True
        
        return is_fitness

    #evaluate the fitness
    def evaluate_fitness(self):
        self.fitness = self.process_data()
        
        if  self.fitness_check(self.fitness, self.fit_best):
            self.pos_best  = self.position
            self.fit_best = self.fitness
            
        #print("fitness")
        #print(self.fitness)
    
    def update_velocity(self, pos_best_global):
        c1 = 1
        c2 = 2
        w  = 0.5
        
        for i in range(0, self.f_count):
            
            r1 = np.random.uniform(low=-1, high=1, size=1)[0]
            r2 = np.random.uniform(low=-1, high=1, size=1)[0]
            
            velocity_cog = c1*r1*(self.pos_best[i]-self.position[i])
            velocity_soc = c2*r2*(pos_best_global[i]-self.position[i])
            
            self.velocity[i]=w*self.velocity[i]+velocity_cog+velocity_soc
    
    def update_position(self):
        for i in range(0, self.f_count):
            self.pos_act[i] = self.pos_act[i] + self.velocity[i]
            
            #adjust max value 
            if self.pos_act[i] > 1:
                self.pos_act[i] = 0.9
            
            if self.pos_act[i] < 0 :
                self.pos_act[i] = 0.0
                
            self.position[i] = 1 if self.pos_act[i] > 0.5 else 0       
        
    def print_position(self):
        print(self.position)
    
    def print_velocity(self):
        print(self.velocity)