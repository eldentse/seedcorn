import numpy as np
import pandas as pd

def count_unique_policy_num(df):
  """
  Wraper for computing number of unique policy

  Args:
    df: can be any dataframe
  Returns:
    no. of unique policy (int)

  """
  return df['CHDRNUM'].nunique()


def to_categorical(y, num_classes=None, dtype="float32"):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def to_lookup(lookup_column, value_column):
  """Create unqiue lookup table from df columns"""
  name_to_idx = {}
  counter = 0
  if lookup_column is None:
    return {'regression': None}, np.expand_dims(value_column, axis=-1)
  
  if len(lookup_column) != len(value_column):
        print('Data is binned')
        for value in value_column:
          name_to_idx[lookup_column[value]] = value
  
  else:
    for i, name in enumerate(lookup_column):
          if not (name) in name_to_idx.keys():
            name_to_idx[name] = counter
            value_column[i] = counter
            counter += 1
          else:
            value_column[i] = name_to_idx[name]
  return name_to_idx, value_column


def unique_class_idx(label_column):
  """Return the indices of unique classes"""
  temp = to_categorical(label_column)
  return np.where(np.sum(temp, axis=0) == 1)[0]


def filename_remap(lookup_values, y):
  """Group filaname class"""
  mapping = {'ZSULINK':'Link',
             'ZSTERM':'Term',
             'ZSPAR':'Par',
             'ZSANN':'Ann',
             'ZSNONPAR':'NonPar',
             'ZSPARANN':'Ann',
             'ZSPARJ':'Par',
             'ZBPAR':'Par',
             'ZBNONPAR':'NonPar',
             'ZBCBP':'BP',
             'ZBCBP2':'BP',
             'ZBTERM':'Term',
             'ZSCBP':'BP',
             'ZSCBP2':'BP'
             }

  name_to_idx = {'Link'  :0,
                 'Term'  :1, 
                 'Par'   :2, 
                 'Ann'   :3, 
                 'NonPar':4, 
                 'BP'    :5
                 }
  
  for i, name in enumerate(lookup_values):
        lookup_values[i] = mapping[name]
        y[i] = name_to_idx[mapping[name]]
  return lookup_values, y


def to_bin_column(data, num_bins, binning_method):
  """
  Bin a DataFrame column into custom bins using either equal-width or equal-frequency binning.

  Args:
      data (pandas.Series): The column data to be binned.
      num_bins (int): The number of bins to create.
      binning_method (str): The binning method to use, either 'equal-width' or 'equal-frequency'.

  Returns:
      pandas.Series: The binned column data.

  Raises:
      ValueError: If an invalid `binning_method` is provided.
  """
  if binning_method == 'equal-width':
      # Equal-width binning
      return pd.cut(data, num_bins)
  elif binning_method == 'equal-frequency':
      # Equal-frequency binning
      return pd.qcut(data, num_bins, duplicates='drop')
  else:
      raise ValueError("Invalid binning method. Please choose 'equal-width' or 'equal-frequency'.")