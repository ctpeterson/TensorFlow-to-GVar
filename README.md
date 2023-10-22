# TensorFlow-to-GVar
Simple Python code for converting a TensorFlow model into a GVar function. First, create a TensorFlow model using Keras.
```
import keras
layers = layers = [
    keras.layers.Input(shape = (1,)),
    keras.layers.Dense(2, activation = "relu"),
    keras.layers.Dense(2, activation = "relu"),
    keras.layers.Dense(1, activation = "relu")
]; model = keras.Sequential(layers); model.build();
```
Now convert the model function into a GVar function.
```
gvar_model = convert_fixed_input(model)
```
The `gvar_model` function that is returned by `convert_fixed_input` can now be evaluated at a point `x`. For example, 
```
import gvar
print(gvar_model([3.], [gvar.gvar(1., 0.) for p in range(13)]))
```
will produce `[19(0)]`. The argument in `range` is the number of "trainable" parameters that belong to the Keras/TensorFlow model that we created above.
