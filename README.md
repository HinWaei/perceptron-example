# Perceptron-example
A simple linear perceptron 

```Python
import numpy as np

def sign(w,x,b):
  return w*x+b

def train(w,X,b,times,labels,alpha):
  for _ in range(times):
    for input,label in zip(X,labels):
      prediction=sign(w,input,b)
      w+=alpha*(label-prediction)*input
      b+=alpha*(label-prediction)
      yield w,b

'''
for w,b in train(3,[1,2,3,4,5,6,7,8,9,10],2,1000,[4,1,2,5,3,2,2,4,5,6,1],0.04):
  print(w,b)
'''
```
