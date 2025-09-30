import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform()

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),                                  
    tf.keras.layers.Dense(y_onehot.shape[1], activation='softmax')                
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=120, batch_size=5, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.plot(history.history['loss'])
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

def predict(sample):
    sample = np.array(sample).reshape(1, -1).astype(np.float32)  
    probabilities = model.predict(sample)
    predicted_class = np.argmax(probabilities)
    return encoder.categories_[0][predicted_class]

manual_input = [5.1, 3.5, 1.4, 0.2]  

predicted_class = predict(manual_input)
print(f'Predicted class for input {manual_input}: {predicted_class}')
