import pandas as pd
import chess, chess.pgn
import io
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

def fen_to_bitmap_and_who_move_data(fen):
    # Функция переводит fen запись в понятную для нейросети форму
    # Фигура помечается знаком 0, когда ее нет на этом поле, знаком 1, когда она принадлежит игроку, 
    # который должен сделать ход, и знаком −1, когда она принадлежит противнику.

    board = chess.Board()
    board.set_fen(fen)

    bitmap = []
    colors = [chess.WHITE, chess.BLACK]
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for color in colors:
        for piece in pieces:
            squares = board.pieces(piece, color)
            bitmap_ = [0] * 64

            for square in squares:
                if (color == board.turn):
                    bitmap_[square] = 1
                else:
                    bitmap_[square] = -1

            bitmap.extend(bitmap_)
    
    return bitmap

def get_value_from_mate_value(mate_value):
    abs_max_value = 15319
    d_centipawn = 10
    
    res = ((25 - int(mate_value[2:])) * d_centipawn) + abs_max_value
    
    if (mate_value[1] == '-'):
        return res * -1
    return res

def stockfish_value_to_number_value(value):
    if value[0] == '+':
        return int(value[1:])
    if value[0] == '-' or value[0] == '0':
        return int(value)
    if value[0] == '#':
        return get_value_from_mate_value(value)
    
    raise Exception("Dont know start char for value - " + value)
    
def get_values_from_df(df):
    X = []
    y = []
    
    for index, row in df.iterrows():
        X.append(fen_to_bitmap_and_who_move_data(row['FEN']))
        y.append(stockfish_value_to_number_value(row['Evaluation']))
        if (index % 100000 == 0 and index != 0):
            print(index)
        if ((index + 1) % 10000 == 0): # Слишком много данных
            break
    
    return X, y

import sklearn

df = pd.read_csv("deep22_with_trim_fen.csv")

X, y = get_values_from_df(df)
y = sklearn.preprocessing.normalize([np.array(y)]).tolist()[0]

# https://www.researchgate.net/publication/322539902_Learning_to_Evaluate_Chess_Positions_with_Deep_Neural_Networks_and_Limited_Lookahead

from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=(768,)))
 
model.add(Dense(2048, activation='elu'))
model.add(Dense(2048, activation='elu'))
model.add(Dense(2048, activation='elu'))
model.add(Dense(1, activation='linear'))

model.summary()

sgd = keras.optimizers.legacy.SGD(learning_rate=0.001, decay=1e-8, momentum=0.7, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

history = model.fit(X_train, y_train, batch_size=256, epochs=50)
tests = model.evaluate(np.array(X_test), np.array(y_test), batch_size=256)

model.save('16_model_2.h5')
