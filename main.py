import numpy as np
import pandas as pd


# import bicep_tracker
# import pushup_tracker
# import squat_tracker
# import plank_tracker



print("Select Option")
print("1.Bicep")
print("2.Pushups")
print("3.squat")
print("4.plank")
n = int(input())



def switch(n):
    if(n==1):
        exec(open("bicep_tracker.py").read())
    elif(n==2):
        exec(open("pushup_tracker.py").read())
    elif(n==3):
        exec(open("squat_tracker.py").read())
    elif(n==4):
        exec(open("plank_tracker.py").read())




switch(n)





