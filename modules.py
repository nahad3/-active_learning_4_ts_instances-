import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__(in_features, out_features, bias=bias)

  def forward(self, x, params=None, episode=None):
    if params is None:
      x = super(Linear, self).forward(x)
    else:
      weight, bias = params.get('weight'), params.get('bias')
      if weight is None:
        weight = self.weight
      if bias is None:
        bias = self.bias
      x = F.linear(x, weight, bias)
    return x

def get_child_dict(params, key=None):
  """
  Constructs parameter dictionary for a network module.

  Args:
    params (dict): a parent dictionary of named parameters.
    key (str, optional): a key that specifies the root of the child dictionary.

  Returns:
    child_dict (dict): a child dictionary of model parameters.
  """
  if params is None:
    return None
  if key is None or (isinstance(key, str) and key == ''):
    return params

  key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
  if not any(filter(key_re.match, params.keys())):  # handles nn.DataParallel
    key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))
  child_dict = OrderedDict(
    (key_re.sub(r'\1', k), value) for (k, value)
      in params.items() if key_re.match(k) is not None)
  return child_dict


class Sequential(nn.Sequential):
  def __init__(self, *args):
    super(Sequential, self).__init__(*args)

  def forward(self, x, params=None, episode=None):
    if params is None:
      for module in self:
        x = module(x, None, episode)
    else:
      for name, module in self._modules.items():
        if "relu" in name.lower():
          x = module(x)
        else:
          x = module(x, get_child_dict(params, name))
    return x


class Conv1D(nn.Conv1d):
  'custom conv1D function that takes in parameters'
  def __init__(self, in_channels,out_channels,kernel_size,**kwargs):

    super(Conv1D, self).__init__(in_channels, out_channels, kernel_size,**kwargs)
    self.kernel_size = kernel_size
    if kwargs.get('padding') == None:
      self.padding = 0
    else:
      self.padding = kwargs['padding']
    if kwargs.get('dilation') == None:
      self.dilation = 1
    else:
      self.dilation = kwargs.get('dilation')
    if kwargs.get('groups') == None:
      self.groups = 1
    else:
      self.groups = kwargs.get('groups')
    if kwargs.get('stride') == None:
      self.stride = 1
    else:
      self.stride = kwargs.get('stride')

  def forward(self, x, params=None):
    if params is None:
      x = super(Conv1D, self).forward(x)
    else:
      weight, bias = params.get('conv.weight'), params.get('conv.bias')
      if weight is None:
        weight = self.weight
      if bias is None:
        bias = self.bias
      x = F.conv1d(x,weight=weight,bias=bias,stride= self.stride, padding=self.padding,dilation=self.dilation,\
                   groups=self.groups)
    return x
