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
    return 100.0 * np.abs((model_val - reference_val) / reference
             
