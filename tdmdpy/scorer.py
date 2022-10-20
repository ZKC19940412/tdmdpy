from .ffscore import experimental_reference, tolerance, units
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

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

    return int(final_score)

def score_property(modelled_val, property_indices=0):
    """Score static property of water using the score function
           from Carlos Vega et al and percent error
           DOI: 10.1039/c1cp22168j
           input:
           modelled_val: (float) predicted property from ML md
           property_indices: (int) indices of the property
    """
    # Extract needed information
    experimental_reference_val = list(experimental_reference.values())[property_indices]
    ffscore_tolerance = list(tolerance.values())[property_indices]

    # Compute ff score and percent error
    ffscore = force_field_score_scheme(modelled_val, experimental_reference_val, ffscore_tolerance)
    percent_error = mean_absolute_percentage_error([experimental_reference_val], [modelled_val]) * 100

    # Write information into dictionary
    summary_dictionary = {'Property': list(experimental_reference.keys())[property_indices],
                          'Units': list(units.values())[property_indices],
                          'FF Score Tolerance (%)': ffscore_tolerance,
                          'FF Score ': ffscore,
                          'Experimental reference': experimental_reference_val,
                          'Modelled': modelled_val,
                          'Percent error (%)': np.round(percent_error, 2)}

    # Pretty print
    n = 44
    print(n * "=")
    print(f"{'              Score Summary       ':{n}s}")
    print(n * "=")
    for i in range(2):
        print(f"{'              %s       ':{n}s}" % (list(summary_dictionary.keys())[i] + " : " + list(
            summary_dictionary.values())[i]))
    for j in range(2, 7):
        print(f"{'              %s       ':{n}s}" % (list(summary_dictionary.keys())[j] + " : " + str(list(
            summary_dictionary.values())[j])))

    print(n * "=")
    np.save('summary.npy', summary_dictionary)
