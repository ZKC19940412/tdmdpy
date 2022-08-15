from .ffscore import experimental_reference
from .ffscore import tolerance

def force_field_score_scheme(prediction, ground_truth, tolerance):
    """Score static property of water using the score function
       from Carlos Vega et al.
       DOI: 10.1039/c1cp22168j
       input:
       prediction: (float) predicted property from snap md
       ground_truth: (float) experimental properties
       tolerance: (float) tolerance factor in %
       property_str: (str) name of predicted property
       output:
       final score: (int) final score of the property,
    """
    
    base_score = np.round(10 - np.abs(100 * (
            prediction - ground_truth) / (ground_truth * tolerance)))
    final_score = np.max([base_score, 0])

    return final_score

def compute_normal_percent_error(reference_val, model_val):
    """Compute percent error based on scalar quantity
           input:
           reference_val: (float) reference value
           model_val: (float) modelled value

           output:
           Percent error with respect to reference
    """
    return 100.0 * np.abs((model_val - reference_val) / reference)

def compute_matching_score(reference_arr, modelled_arr):
    """Compute matching score between two arrays
               input:
               reference_array: (ndarray) reference array
               model_array: (ndarray) modelled array

               output:
               matching score between two array 0 ~ 100 scale

    """
    absolute_difference = np.abs(reference_arr - modelled_arr)
    mae = np.sum(absolute_difference / (np.sum(reference_arr) + np.sum(modelled_arr)))
    matching_score = (1 - mae) * 100
    return matching_score


def compute_normal_percent_error(reference_val, model_val):
    """Compute percent error based on scalar quantity
           input:
           reference_val: (float) reference value
           model_val: (float) modelled value

           output:
           Percent error with respect to reference
    """
    return 100.0 * np.abs((model_val - reference_val) / reference_val)


def interpolate_array(x, y, new_x_size):
    """interpolate array to account for difference in sizes
               input:
               x: (ndarray) x-scale 
               y: (ndarray) y-scale
               new_x_size: (int) size of the new_x_array

               output:
               new_x, new_y: x and y-scale with new sizes
    """
    # Construct new x based on its size
    new_x = np.linspace(x.min(), x.max(), new_x_size)

    # Construct new y based on 1D interpolation
    new_y = np.interp(new_x, x, y)

    return new_x, new_y
             
