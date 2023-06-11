import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Any
from matplotlib.patches import Patch



def preprocess(
        df: pd.DataFrame,
        hour: int,
        hours_before: int,
        valid: bool
    ) -> List[torch.tensor]:

    """
    Preprocesa los datos del csv.
    
    Parameters
    ----------
    df           : Dataframe con los datos.
    hour         : Hora de la que queremos los datos.
    hours_before : Numero de horas previas que se utilizan para predecir.
    valid        : Si se quiere conjunto de validacion o no.
    
    Returns
    -------
    out          : Conjuntos y labels de train, test y validacion y la curva
                   de la misma hora el dia anterior para cada curva.
    """

    dates = list(df['date'].values)
    supplies = list(df['supply'].values)
    weekdays = list(df['weekday'].values)
    months = list(df['month'].values)
    
    # Creacion inicial de los conjuntos
    X = []
    Y_prev = []
    Y = []
    vectores = [] # para guardar las variables exogenas
    for i in range(0, len(dates) - hours_before, hours_before):
        target = i + hours_before + hour
        # Datos de las HOURS_BEFORE horas anteriores
        vec_supplies = []
        for s in supplies[i:i+hours_before]:
            vec_supplies += s
        # Dia de la semana
        vec_weekday = [0 for _ in range(3)]
        if weekdays[target] == 5: # sabado
            vec_weekday[1] = 1
        elif weekdays[target] == 6: # domingo
            vec_weekday[2] = 1
        else: # entre semana
            vec_weekday[0] = 1
        # Estacion
        month = months[target]
        vec_season = [0 for _ in range(4)]
        if 2 <= month <= 4:
            season = 0
        elif 5 <= month <= 7:
            season = 1
        elif 8 <= month <= 10:
            season = 2
        else:
            season = 3
        vec_season[season] = 1
        # Juntamos todo
        vectores.append(vec_weekday + vec_season)
        # X e Y
        X.append(vec_supplies)
        Y_prev.append(supplies[target-24])
        Y.append(supplies[target])
    
    # Train, validacion y test
    if valid:
        frac_train = 0.6
        frac_valid = 0.2
    else:
        frac_train = 0.8
    n_train = int(frac_train*len(X))
    if valid:
        n_valid = int(frac_valid*len(X))  
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    if valid:
        X_valid = X[n_train:n_train+n_valid]
        Y_valid = Y[n_train:n_train+n_valid]
        X_test = X[n_train+n_valid:]
        Y_test = Y[n_train+n_valid:]
        Y_prev = Y_prev[n_train+n_valid:]
    else:
        X_test = X[n_train:]
        Y_test = Y[n_train:]
        Y_prev = Y_prev[n_train:]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    if valid:
        X_valid = np.array(X_valid)
        Y_valid = np.array(Y_valid)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_prev = np.array(Y_prev)
    
    # Normalizacion
    min_x = np.min(X_train)
    max_x = np.max(X_train)
    X_train = (X_train - min_x) / (max_x - min_x)
    Y_train = (Y_train - min_x) / (max_x - min_x)
    if valid:
        X_valid = (X_valid - min_x) / (max_x - min_x)
        Y_valid = (Y_valid - min_x) / (max_x - min_x)
    X_test = (X_test - min_x) / (max_x - min_x)
    Y_test = (Y_test - min_x) / (max_x - min_x)
    
    # Concatenacion con las variables exogenas
    X_tr = []
    X_va = []
    X_te = []
    for i, v in enumerate(vectores):
        if i < len(X_train):
            X_tr.append(list(X_train[i]) + v)
        else:
            if valid:
                if i < len(X_train) + len(X_valid):
                    X_va.append(list(X_valid[i-len(X_train)]) + v)
                else:
                    X_te.append(list(X_test[i-len(X_train)-len(X_valid)]) + v)
            else:
                X_te.append(list(X_test[i-len(X_train)]) + v)
    X_train = np.array(X_tr)
    if valid:
        X_valid = np.array(X_va)
    X_test = np.array(X_te)
    
    # Conversion a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    if valid:
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        Y_valid = torch.tensor(Y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    
    # Output
    if valid:
        out = [X_train, Y_train, X_valid, Y_valid, X_test, Y_test, Y_prev]
    else:
        out = [X_train, Y_train, X_test, Y_test, Y_prev]
        
    return out



def calculate_integral(
        real_curve_price: List[float],
        real_curve_supply: List[float],
        approx_curve_price: List[float],
        approx_curve_supply: List[float]
    ) -> float:
    
    """
    Calcula la integral de las curvas escalonadas.
    
    Parameters
    ----------
    real_curve_price    : Precios de la curva real.
    real_curve_supply   : Ofertas de la curva real.
    approx_curve_price  : Precios de la curva aproximada.
    approx_curve_supply : Ofertas de la curva aproximada.
    
    Returns
    -------
    integral            : Valor de la integral.
    """
  
    # Diccionario con todos los precios ordenados de menor a mayor. Cada precio
    # tiene asociada una lista que dice si el precio era real o de la
    # aproximacion y tambien contiene la oferta asociada
    if real_curve_price[0] == 0:
        real_curve_price[0] = 10**(-6)
    all_prices = list(sorted(real_curve_price + approx_curve_price))
    all_prices_dict = {}
    i_real = 0
    i_approx = 0
    prices_seen = []
    for i, price in enumerate(all_prices):
        if price in real_curve_price and price not in prices_seen:
            all_prices_dict[i] = [price, 'real', real_curve_supply[i_real]]
            i_real += 1
        elif price in approx_curve_price:
            all_prices_dict[i] = [price, 'approx',
                                  approx_curve_supply[i_approx]]
            i_approx += 1
        prices_seen.append(price)
    
    
    # Para guardar ofertas reales y aproximadas
    s_real = []
    s_approx = []

    # EL primer dato siempre sera de la curva aproximada
    last_real = 0
    last_approx = approx_curve_supply[0]

    for i, p in enumerate(all_prices[1:]):
        type_p, s = all_prices_dict[i][1], all_prices_dict[i][2]
        if type_p == 'real':
            s_real.append(s)
            last_real = s
            s_approx.append(last_approx)
        else:
            s_approx.append(s)
            last_approx = s
            s_real.append(last_real)
    
    # Calculo de la integral
    integral = 0
    i = 0
    for real, approx in zip(s_real, s_approx):
        base = all_prices[i+1] - all_prices[i]
        altura = real - approx
        integral += abs(base*altura)
        i += 1
    
    return integral



def view_pred(
        xx: np.array,
        Y_test: np.array,
        Y_prev: np.array,
        Y_pred_RF: np.array,
        Y_pred_LSTM: np.array,
        Y_pred_LSTM_hidden: np.array,
        cmap: Any,
        idx: Any = None
    ) -> None:
    
    """
    Dibuja las predicciones de un dia aleatorio del test.
    
    Parameters
    ----------
    xx                 : Precios de la malla.
    Y_test             : Curvas reales del test.
    Y_prev             : Prediccion heuristica.
    Y_pred_RF          : Predicciones del Random Forest.
    Y_pred_LSTM        : Predicciones de la LSTM.
    Y_pred_LSTM_hidden : Predicciones de la LSTM con hidden state.
    cmap               : Mapa de colores.
    idx                : Indice del dia de entrenamiento.
    """

    if idx is None:
        idx = np.random.randint(0, len(Y_test))

    fig, ax = plt.subplots(figsize=(10, 6))

    legend_elements = [Patch(facecolor=cmap(0/4), label='Real'),
                       Patch(facecolor=cmap(1/4), label='Día anterior'),
                       Patch(facecolor=cmap(2/4), label='Random Forest'),
                       Patch(facecolor=cmap(3/4), label='LSTM'),
                       Patch(facecolor=cmap(4/4), label='LSTM hidden')]

    linewitdh = 2

    ax.step(xx, list(Y_test[idx]) + [list(Y_test[idx])[-1]], color=cmap(0/4),
            where='post', linewidth=linewitdh)
    ax.step(xx, list(Y_prev[idx]) + [list(Y_prev[idx])[-1]], color=cmap(1/4),
            where='post', linewidth=linewitdh)
    ax.step(xx, list(Y_pred_RF[idx]) + [list(Y_pred_RF[idx])[-1]],
            color=cmap(2/4), where='post', linewidth=linewitdh)
    ax.step(xx, list(Y_pred_LSTM[idx]) + [list(Y_pred_LSTM[idx])[-1]],
            color=cmap(3/4), where='post', linewidth=linewitdh)
    ax.step(xx, list(Y_pred_LSTM_hidden[idx]) +
            [list(Y_pred_LSTM_hidden[idx])[-1]], color=cmap(4/4),
            where='post', linewidth=linewitdh)
    
    ax.grid()
    ax.set_xlabel('Precio (€)')
    ax.set_ylabel('Oferta (kWh)')
    ax.set_title('Prediccion de un dia aleatorio del test')
    ax.legend(handles=legend_elements)