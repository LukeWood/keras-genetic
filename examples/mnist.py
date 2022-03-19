from tensorflow import keras
import keras_genetic

(x_train,  y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(metrics=['accuracy'])

def evaluate_accuracy(individual: keras_genetic.Individual):
    model = individual.load_model()
    result = model.evaluate(x_train, y_train, return_dict=True)
    return result['accuracy']

results = keras_genetic.search(
    model=model,
    # computational cost is evaluate*generations*population_size
    evaluator=evaluate_accuracy,
    generations=25,
    population_size=25,
    breeder=keras_genetic.breeder.RandomFeatureMutationBreeder()
    return_best=1,
)

result_model = results.best.load_model()

result = model.evaluate(x_test, y_test, return_dict=True)
print("Accuracy:", result['accuracy'])
