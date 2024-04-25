
from typing import Any

class EarlyStopping:
    def __init__(self,):
        self.counter = 0
        self.early_stop = False
        self.lr_decreases = 0
        self.best_val_loss = float('inf')

    def __call__(self, validation_loss) -> Any:
        
        if validation_loss < self.best_val_loss:
            self.best_val_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter}')

        if self.counter >= 5:
            if self.lr_decreases >= 2:
                self.early_stop = True

            else: 
                print('Early stopping criterion met, decreasing learning rate')
                self.lr_decreases +=1
                self.early_stop = False
