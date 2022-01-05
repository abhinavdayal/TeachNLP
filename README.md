# TeachNLP
A From Scratch NLP teaching utilities and concepts.
## Vensor_flow
* Custom library to implement our own tensor kind of data structure named Vensor that supports back propagation.
* Steps to use Vensor data structure  

1. Clone this repo 
2. Add the path to TeachNLP to the PATH variable 
       or 
  run these two instructions in python 
  ```
       import sys 
       sys.path.append(path_to_TeachNLP)
  ```

3. Vensor can be imported using 
       ```from vensor_flow impor Vensor```


### Operations on Vensor 
* Vensor Constructor 
```
  Vensor(data,requires_grad=False, is_leaf=True) 
  Parameters 
       data : array_like object 

             Any object exposing the array interface
             An object whose __array__ method returns an array,
             or any (nested) sequence. If object is a scalar,
            a 0-dimensional array containing object is returned
       requires_grad : A boolean value specifying 
             wheather to record computation graph or not 
        is_leaf : A boolean value specifying wheather
             current Vensor is leaf or not
       
```
