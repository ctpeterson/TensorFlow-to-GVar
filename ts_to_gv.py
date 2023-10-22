import tensorflow as _tf # TensorFlow
import numpy as _np # NumPy
import gvar as _gv # GVar

@_tf.function
def output_with_jac(x, model):
    with _tf.GradientTape() as tp: model_output = model(x)
    return model_output, tp.gradient(model_output, model.trainable_variables)

@_tf.function
def output_without_jac(x, model):
    return model(x)

def convert_fixed_input(model):
    # For collecting model parameters
    variables = [var for var in model.trainable_variables]
    shapes = [_np.shape(_np.array(var.value())) for var in variables]
    lngths = [0] + list(_np.cumsum([
        _np.prod([*shapes[ind]]) for ind, var in enumerate(variables)
    ]));
            
    # Create GVar function
    def gvar_model(x, p, set_weights = True):
        if set_weights: model.set_weights(
                [_gv.mean(_np.array(p[lngths[ind]:lngths[ind+1]]).reshape(shapes[ind]))
                 for ind, var in enumerate(variables)]
        ); x = x if hasattr(x, "__len__") else [x];
        if isinstance(p[0], _gv.GVar):
            model_output, model_jacobian = output_with_jac(_tf.constant(x), model);
            return [_gv.gvar_function(
                p, model_output,
                [jv for mj in model_jacobian for jv in _np.array(mj).flatten()]
            )]
        else: return [out for owj in output_without_jac(_tf.constant(x), model)
                      for out in _np.array(owj).flatten()]

    # Return GVar function
    return gvar_model
