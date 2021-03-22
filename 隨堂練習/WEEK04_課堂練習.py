# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Ramanujan's Formula for Pi

#######################################
#coeff=
#summ=
#k=

#for iter in list[]:
#    denominator=
#    fraction=
#    temp=coeff*fraction/denominator
#    summ
       
#rpi=
#######################################

import math

def factorial(x):
    if x<=1:
        return 1
    return x*factorial(x-1)

def pi_formula():
    k=0
    sum=0
    coeff=(math.sqrt(8))/9801
    
    for k in range(0,10):
        numerator=(factorial(4*k)*(1103+26390*k))
        denominator=(math.pow(factorial(k),4)*math.pow(396,4*k))
        sum+=coeff*numerator/denominator
        
    pi=1/sum
        
    return(pi)
    
print("Pi value using Ramanujan Formula : ",pi_formula())
    