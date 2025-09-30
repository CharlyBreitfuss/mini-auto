import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sensorwerte (x_train) – links, mitte, rechts
x_train = np.array([
    [0.1, 0.9, 0.1],
    [0.0, 0.2, 0.9],
    [0.9, 0.1, 0.0],
    [0.3, 0.7, 0.3],
    [0.2, 0.4, 0.9],
    [0.5, 0.9, 0.5],
    [0.0, 0.0, 0.9],
    [0.9, 0.0, 0.0],
    [0.2, 0.6, 0.8],
    [0.8, 0.2, 0.7],
    [0.1, 0.5, 0.9],
    [0.4, 0.4, 0.6],
    [0.7, 0.9, 0.1],
    [0.0, 0.8, 0.0],
    [0.3, 0.3, 0.3]
])

# Richtungen (y_train) – 0 = gerade, 1 = links, 2 = rechts
y_train = np.array([0, 2, 1, 0, 2, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0])


# Modell erstellen
model = Sequential([
    Dense(8, input_shape=(3,), activation='relu'),  # 8 Neuronen, Eingabe = 3 Sensorwerte
    Dense(8, activation='relu'),                     # versteckte Schicht
    Dense(3, activation='softmax')                  # 3 Ausgaben: geradeaus, links, rechts
])

# Kompilieren: loss = Verlustfunktion, optimizer = wie das Netz lernt
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trainieren
model.fit(x_train, y_train, epochs=50, verbose=1)

# Vorhersagen testen
print("Vorhersage für [0.1, 0.9, 0.1]:", np.argmax(model.predict(np.array([[0.1, 0.9, 0.1]]))))
print("Vorhersage für [0.0, 0.2, 0.9]:", np.argmax(model.predict(np.array([[0.0, 0.2, 0.9]]))))
