import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from typing import List, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)



# Constantes

N = 60
HOURS_BEFORE = 24
min_x = 16916.4
max_x = 122681.6



class EarlyStoppingHidden:
    
    
    """ Clase para hacer Early Stopping. """


    def __init__(
            self,
            patience: int,
            path: str
        ) -> None:
        
        """
        Constructor de la clase.
        
        Parameters
        ----------
        patience : Numero de epocas permitidas sin mejorar.
        path     : Ruta para guardar los pesos del modelo.
        """

        self.patience = patience
        self.counter = 0 # para contar las epocas que llevamos sin mejorar
        self.best_score = np.Inf # mejor valor conseguido
        self.early_stop = False # flag para saber cuando parar
        self.path = path


    def __call__(
            self,
            val_loss: float,
            model: Any
        ) -> None:
        
        """
        Guarda los pesos del modelo y actualiza el mejor valor y contador si
        se mejora y en caso contrario aumenta el contador y para si procede.
        
        Parameters
        ----------
        val_loss : Nuevo valor de la funcion de perdida.
        model    : Modelo al que se le aplica Early Stopping.
        """

        score = val_loss

        if score >= self.best_score: # no mejoramos
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else: # mejoramos
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(
            self,
            val_loss: float,
            model: Any
        ) -> None:
        
        """
        Guarda el modelo y actualiza el mejor valor de la funcion de perdida.
        
        Parameters
        ----------
        val_loss : Nuevo valor de la funcion de perdida.
        model    : Modelo al que se le aplica Early Stopping.
        """

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
        
class LSTM_hidden(nn.Module):
    
    
    """ Clase LSTM. """

    
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            hidden_1: int,
            hidden_2: int,
            hidden_3: int,
            hidden_4: int,
            n_labels: int,
            p: float
        ) -> None:
        
        """
        Constructor de la clase.
        
        Parameters
        ----------
        input_size  : Dimension de entrada de la LSTM.
        hidden_size : Dimension de salida de la LSTM.
        hidden_i    : Dimensiones de la capa i de la MLP.
        n_labels    : Dimension de salida de la MLP.
        p           : Probabilidad de retencion del Dropout.
        """

        super().__init__()
        
        self.exog = input_size > N*HOURS_BEFORE
        self.lstm = nn.LSTM(N*HOURS_BEFORE, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size + (input_size - N*HOURS_BEFORE),
                             hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.fc5 = nn.Linear(hidden_4, n_labels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
                                                                
        
    def forward(
            self,
            x: torch.tensor
        ) -> torch.tensor:
        
        """
        Pasa un tensor a traves de la red.
        
        Parameters
        ----------
        x : Tensor de entrada.
        """
        
        # Las primeras N*DAYS_BEFORE entradas seran las curvas del dia anterior
        if self.exog:
            lstm_input = x[:, :N*HOURS_BEFORE]
            exog = x[:, N*HOURS_BEFORE:]
        else:
            lstm_input = x[:, :N*HOURS_BEFORE]

        # LSTM
        x, (h, c) = self.lstm(lstm_input)
        
        # Hidden State
        hidden = h.clone()
        for _ in range(x.shape[0]-1):
            hidden = torch.cat((hidden, h), dim=0)
        x = torch.cat((x, hidden), dim=1)

        # Concatenamos variables exogenas
        if self.exog:
            x = torch.cat((x, exog), dim=1)

        # MLP
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        
        # Sigmoide
        x = self.sigmoid(x)

        return x
    
    
    
class LSTM_hidden_extended(LSTM_hidden):
    
    
    """ Extension de la clase LSTM. """


    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            hidden_1: int,
            hidden_2: int,
            hidden_3: int,
            hidden_4: int,
            n_labels: int,
            p: float,
            patience: int,
            epochs: int = 50,
            valid: bool = False,
            batch_size: int = 128,
            lr: float = 0.001,
            print_every: int = 5,
            path: str = 'LSTM_hidden.pt'
        ) -> None:
        
        """
        Parameters
        ----------
        input_size  : Dimension de entrada de la LSTM.
        hidden_size : Dimension de salida de la LSTM.
        hidden_i    : Dimensiones de la capa i de la MLP.
        n_labels    : Dimension de salida de la MLP.
        p           : Probabilidad de retencion del Dropout.
        patience    : Numero de epocas permitidas sin mejorar en la funcion de
                      perdida del conjunto de validacion.
        epochs      : Numero de epocas de entrenamiento.
        valid       : Flag que indica si se usa conjunto de validacion o no.
        batch_size  : Batch size.
        lr          : Learning rate.
        print_every : Numero de epocas para imprimir por pantalla la evolucion
                      del entrenamiento.
        path        : Ruta para guardar los pesos del modelo.
        """
        
        super().__init__(
            input_size,
            hidden_size,
            hidden_1,
            hidden_2,
            hidden_3,
            hidden_4,
            n_labels,
            p
        )
        
        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.epochs = epochs
        self.valid = valid
        self.batch_size = batch_size
        self.print_every = print_every
        self.path = path
        self.early_stopping = EarlyStoppingHidden(patience, self.path)
        
        # GPU
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.to(self.device)
        
        # Funcion de perdida
        self.criterion = nn.MSELoss()
        
        # Listas para guardar las metricas durante el entrenamiento
        self.train_loss = []
        self.valid_loss = []
        self.train_rmse = []
        self.train_mae = []
        self.train_mape = []
        self.train_r2 = []
        self.valid_rmse = []
        self.valid_mae = []
        self.valid_mape = []
        self.valid_r2 = []                                              
    

    def trainloop(
            self,
            x_train: torch.tensor,
            y_train: torch.tensor,
            x_valid: torch.tensor,
            y_valid: torch.tensor
        ) -> None:
        
        """
        Entrenamiento de la red.
        
        Parameters
        ----------
        x_train : Conjunto de entrenamiento.
        y_train : Labels del conjunto de entrenamiento.
        x_valid : Conjunto de validacion.
        y_valid : Labels del conjunto de validacion.
        """

        self.num_train = len(x_train)
        self.num_batchs_train = self.num_train // self.batch_size
        if self.valid:
            self.num_valid = len(x_valid)
            self.num_batchs_valid = self.num_valid // self.batch_size
        
        for e in range(1, self.epochs+1):
            
            # Entrenamiento
            self.train()
            running_loss = 0
            idx_train = np.random.permutation(self.num_train)
            for i in range(self.num_batchs_train):
                idx_batch = idx_train[i*self.batch_size:(i+1)*self.batch_size]
                embeddings = x_train[idx_batch].to(self.device)
                labels = y_train[idx_batch].to(self.device)
                self.optim.zero_grad()
                out = self.forward(embeddings)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
            self.train_loss.append(running_loss/self.num_batchs_train)
            # Metricas de entrenamiento
            y_pred = self.predict(x_train)
            MSE = mean_squared_error(y_train * (max_x-min_x) + min_x, y_pred)
            RMSE = np.sqrt(MSE)
            MAE = mean_absolute_error(y_train * (max_x-min_x) + min_x, y_pred)
            MAPE = mean_absolute_percentage_error(y_train * (max_x-min_x)
                                                  + min_x, y_pred)
            R2 = r2_score(y_train * (max_x-min_x) + min_x, y_pred)
            self.train_rmse.append(RMSE)
            self.train_mae.append(MAE)
            self.train_mape.append(100*MAPE)
            self.train_r2.append(R2)

            # Validacion
            if self.valid:
                self.eval()
                running_loss = 0
                idx_valid = np.random.permutation(self.num_valid)
                with torch.no_grad():
                    for i in range(self.num_batchs_valid):
                        idx_batch = idx_valid[i*self.batch_size :
                                              (i+1)*self.batch_size]
                        embeddings = x_valid[idx_batch].to(self.device)
                        labels = y_valid[idx_batch].to(self.device)
                        out = self.forward(embeddings)
                        loss = self.criterion(out, labels)
                        running_loss += loss.item()
                self.valid_loss.append(running_loss/self.num_batchs_valid)
                # Metricas de validacion
                y_pred = self.predict(x_valid)
                MSE = mean_squared_error(y_valid * (max_x-min_x) + min_x,
                                         y_pred)
                RMSE = np.sqrt(MSE)
                MAE = mean_absolute_error(y_valid * (max_x-min_x) + min_x,
                                          y_pred)
                MAPE = mean_absolute_percentage_error(y_valid * (max_x-min_x)
                                                      + min_x, y_pred)
                R2 = r2_score(y_valid * (max_x-min_x) + min_x, y_pred)
                self.valid_rmse.append(RMSE)
                self.valid_mae.append(MAE)
                self.valid_mape.append(100*MAPE)
                self.valid_r2.append(R2)

            # Actualizacion por pantalla del entrenamiento
            if e % self.print_every == 0 or e == 1:
                print('\nEpoch {}\n'.format(e))
                print('Training loss: {}'.format(self.train_loss[-1]))
                if self.valid:
                    print('Validation loss: {}'.format(self.valid_loss[-1]))
            
            # Early stopping
            if self.valid:
                self.early_stopping(self.valid_loss[-1], self)
                if self.early_stopping.early_stop:
                    print('Early stopping')
                    self.load_state_dict(torch.load(self.path))
                    break


    def predict(
            self,
            x_batch: torch.tensor
        ) -> List[List[float]]:
        
        """
        Predice un conjunto de dias.
        
        Parameters
        ----------
        x_batch : Conjunto de dias a predecir.
        
        Returns
        -------
        preds   : Lista con las predicciones para cada dia.
        """

        self.eval()
        
        with torch.no_grad():
            preds = np.array(self.forward(x_batch).detach().numpy()) * \
                (max_x-min_x) + min_x
        
        return preds


    def show_training(
            self
        ) -> None:
        
        """
        Muestra la evolucion del entrenamiento.
        """

        plt.figure(figsize=(14, 8))
        plt.grid()
        plt.xlabel('Epoch')
        plt.plot(range(1, len(self.train_loss)+1), self.train_loss)
        if self.valid:
            plt.plot(range(1, len(self.valid_loss)+1), self.valid_loss)
            plt.legend(['Train', 'Validation'])

        plt.figure(figsize=(14, 8))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.plot(range(1, len(self.train_rmse)+1), self.train_rmse)
        if self.valid:
            plt.plot(range(1, len(self.valid_rmse)+1), self.valid_rmse)
            plt.legend(['Train', 'Validation'])
        
        plt.figure(figsize=(14, 8))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.plot(range(1, len(self.train_mae)+1), self.train_mae)
        if self.valid:
            plt.plot(range(1, len(self.valid_mae)+1), self.valid_mae)
            plt.legend(['Train', 'Validation'])

        plt.figure(figsize=(14, 8))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        plt.plot(range(1, len(self.train_mape)+1), self.train_mape)
        if self.valid:
            plt.plot(range(1, len(self.valid_mape)+1), self.valid_mape)
            plt.legend(['Train', 'Validation'])
        
        plt.figure(figsize=(14, 8))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('$R^2$')
        plt.plot(range(1, len(self.train_r2)+1), self.train_r2)
        if self.valid:
            plt.plot(range(1, len(self.valid_r2)+1), self.valid_r2)
            plt.legend(['Train', 'Validation'])