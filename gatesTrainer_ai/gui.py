from tkinter import *
import os
import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageTk
from matplotlib import style
from matplotlib import figure
from PIL import ImageTk, Image

import time
root= Tk()
root.title("ai project")
root.geometry("500x700")
root.maxsize(700,600)
root.minsize(400,200)






bg = PhotoImage(file="pic.png")

# Show image using label
label1 = Label(root, image=bg)
label1.place(x=0, y=0)
Font_tuple = ("GRIFON Bold", 20, "bold")
i = IntVar()
def UnitStep(inp):
    if inp >= 0:
        return 1
    else:
        return 0
def AndPerceptron(and_input):
  plt.clf()

  a = [0,0, 1,1]

  b = [0,1,0,1]
  y = [0,0,0,1] # Actual Output

  w = [0.3,-0.1]

  threshold = -0.2

  learning_rate = 0.1
  i=0
  while i<16 :
      boundary_lines = []
      Yd = (a[i%4]*w[0]+b[i%4]*w[1])+threshold
      if w[1]!= 0:
          x2= -threshold/w[1]
      else:
          x2 = 0
      if w[0]!= 0:
          x1= -threshold/w[0]
      else:
          x2 = 0

      output = UnitStep(Yd)
      print("Input : " +str(a[i%4])+ "," + str(b[i%4]))
      print("W : " +str(w[0])+ "," + str(w[1]))
      print("Yd: " +str(Yd))
      print("Actual Output : " +str(y[i%4])+ ",Predicted Output" + str(output))
      e= y[i%4]-output
      w[0]=w[0]+learning_rate*e*a[i%4]
      w[1]=w[1]+learning_rate*e*b[i%4]


      print("New W : " + str(w[0]) + "," + str(w[1]))
      i=i+1
  if w[1] != 0:
      x2 = -threshold / w[1]
  else:
      x2 = 0
  if w[0] != 0:
      x1 = -threshold / w[0]
  else:
      x2 = 0
  plt.rc('figure', figsize=(10, 5))
  plt.style.use('fivethirtyeight')
  X = [0, 0, 1, 1]

  Y = [0, 1, 0, 1]
  color = ['red']

  plt.scatter([0, 0, 1], [0, 1, 0], color='blue')
  plt.scatter([1], [1], color='red')

  plt.xlabel('X input feature')
  plt.ylabel('Y input feature')
  plt.title('Perceptron regression for AND Gate')
  plt.plot()
  plt.tight_layout
  plt.plot([x1, 0], [0, x2], 'k-')
  plt.show()
  Yd = (and_input[0] * w[0] + and_input[1] * w[1]) - threshold
  return UnitStep(Yd)
def ORPerceptron(and_input):
  plt.clf()

  a = [0,0, 1,1]

  b = [0,1,0,1]
  y = [0,1,1,1] # Actual Output

  w = [0.3,-0.1]

  threshold = -0.2

  learning_rate = 0.1
  i=0
  while i<16 :
      boundary_lines = []
      Yd = (a[i%4]*w[0]+b[i%4]*w[1])+threshold
      if w[1]!= 0:
          x2= -threshold/w[1]
      else:
          x2 = 0
      if w[0]!= 0:
          x1= -threshold/w[0]
      else:
          x2 = 0

      output = UnitStep(Yd)
      print("Input : " +str(a[i%4])+ "," + str(b[i%4]))
      print("W : " +str(w[0])+ "," + str(w[1]))
      print("Yd: " +str(Yd))
      print("Actual Output : " +str(y[i%4])+ ",Predicted Output" + str(output))
      e= y[i%4]-output
      w[0]=w[0]+learning_rate*e*a[i%4]
      w[1]=w[1]+learning_rate*e*b[i%4]


      print("New W : " + str(w[0]) + "," + str(w[1]))
      i=i+1
  if w[1] != 0:
      x2 = -threshold / w[1]
  else:
      x2 = 0
  if w[0] != 0:
      x1 = -threshold / w[0]
  else:
      x2 = 0
  plt.rc('figure', figsize=(10, 5))
  plt.style.use('fivethirtyeight')
  X = [0, 0, 1, 1]

  Y = [0, 1, 0, 1]
  color = ['red']

  plt.scatter([1, 0, 1], [1, 1, 0], color='red')
  plt.scatter([0], [0], color='blue')

  plt.xlabel('X input feature')
  plt.ylabel('Y input feature')
  plt.title('Perceptron regression for OR Gate')
  plt.plot()
  plt.tight_layout
  plt.plot([x1, 0], [0, x2], 'k-')
  plt.show()
  Yd = (and_input[0] * w[0] + and_input[1] * w[1]) - threshold
  return UnitStep(Yd)


def NANDPerceptron(and_input):
  plt.clf()

  a = [0,0, 1,1]

  b = [0,1,0,1]
  y = [1,1,1,0] # Actual Output

  w = [0.3,-0.1]

  threshold = 0.7

  learning_rate = 0.1
  i=0
  while i<40 :
      boundary_lines = []
      Yd = (a[i%4]*w[0]+b[i%4]*w[1])+threshold
      if w[1]!= 0:
          x2= -threshold/w[1]
      else:
          x2 = 0
      if w[0]!= 0:
          x1= -threshold/w[0]
      else:
          x2 = 0

      output = UnitStep(Yd)
      print("Input : " +str(a[i%4])+ "," + str(b[i%4]))
      print("W : " +str(w[0])+ "," + str(w[1]))
      print("Yd: " +str(Yd))
      print("Actual Output : " +str(y[i%4])+ ",Predicted Output" + str(output))
      e= y[i%4]-output
      w[0]=w[0]+learning_rate*e*a[i%4]
      w[1]=w[1]+learning_rate*e*b[i%4]


      print("New W : " + str(w[0]) + "," + str(w[1]))
      i=i+1
  if w[1] != 0:
      x2 = -threshold / w[1]
  else:
      x2 = 0
  if w[0] != 0:
      x1 = -threshold / w[0]
  else:
      x2 = 0
  plt.rc('figure', figsize=(10, 5))
  plt.style.use('fivethirtyeight')
  X = [0, 0, 1, 1]

  Y = [0, 1, 0, 1]
  color = ['red']

  plt.scatter([0, 0, 1], [0, 1, 0], color='red')
  plt.scatter([1], [1], color='blue')

  plt.xlabel('X input feature')
  plt.ylabel('Y input feature')
  plt.title('Perceptron regression for NAND Gate')
  plt.plot()
  plt.tight_layout
  plt.plot([x1, 0], [0, x2], 'k-')
  plt.show()
  Yd = (and_input[0] * w[0] + and_input[1] * w[1]) - threshold
  return UnitStep(Yd)

def NOTPerceptron(and_input):
  plt.clf()
  a = [0,1]
  y = [1,0]    # Actual Output

  w = -0.0002

  threshold = 0.1

  learning_rate = 0.1
  i=0
  while i<25 :
      boundary_lines = []
      Yd = (a[i%2]*w+threshold)

      x2= threshold

      if w!= 0:
          x1= -threshold/w
      else:
          x1 = 0

      output = UnitStep(Yd)
      print("Input : " +str(a[i%2]))
      print("W : " +str(w))
      print("Yd: " +str(Yd))
      print("Actual Output : " +str(y[i%2])+ ",Predicted Output" + str(output))
      e= y[i%2]-output
      w=w+learning_rate*e*a[i%2]


      print("New W : " + str(w) )
      i=i+1

  plt.rc('figure', figsize=(10, 5))
  plt.style.use('fivethirtyeight')

  plt.scatter([0], [1], color='red')
  plt.scatter([1], [0], color='blue')

  plt.xlabel('X input feature')
  plt.ylabel('Y input feature')
  plt.title('Perceptron regression for NOT Gate')
  plt.plot()
  plt.tight_layout
  plt.plot([x1, 0], [0, x2], 'k-')
  plt.show()
  Yd = (and_input * w ) - threshold
  return UnitStep(Yd)
def NORPerceptron(and_input):
  plt.clf()
  a = [0,0, 1,1]

  b = [0,1,0,1]
  y = [1,0,0,0] # Actual Output

  w = [0.3,-0.1]

  threshold = 0.2

  learning_rate = 0.1
  i=0
  while i<36 :
      boundary_lines = []
      Yd = (a[i%4]*w[0]+b[i%4]*w[1])+threshold
      if w[1]!= 0:
          x2= -threshold/w[1]
      else:
          x2 = 0
      if w[0]!= 0:
          x1= -threshold/w[0]
      else:
          x2 = 0

      output = UnitStep(Yd)
      print("Input : " +str(a[i%4])+ "," + str(b[i%4]))
      print("W : " +str(w[0])+ "," + str(w[1]))
      print("Yd: " +str(Yd))
      print("Actual Output : " +str(y[i%4])+ ",Predicted Output" + str(output))
      e= y[i%4]-output
      w[0]=w[0]+learning_rate*e*a[i%4]
      w[1]=w[1]+learning_rate*e*b[i%4]


      print("New W : " + str(w[0]) + "," + str(w[1]))
      i=i+1
  if w[1] != 0:
      x2 = -threshold / w[1]
  else:
      x2 = 0
  if w[0] != 0:
      x1 = -threshold / w[0]
  else:
      x2 = 0
  plt.rc('figure', figsize=(10, 5))
  plt.style.use('fivethirtyeight')
  X = [0, 0, 1, 1]

  Y = [0, 1, 0, 1]
  color = ['red']

  plt.scatter([1, 0, 1], [1, 1, 0], color='blue')
  plt.scatter([0], [0], color='red')

  plt.xlabel('X input feature')
  plt.ylabel('Y input feature')
  plt.title('Perceptron regression for NOR Gate')
  plt.plot()
  plt.tight_layout
  plt.plot([x1, 0], [0, x2], 'k-')
  plt.show()
  Yd = (and_input[0] * w[0] + and_input[1] * w[1]) - threshold
  return UnitStep(Yd)
def sigmoid(x):

 return 1 / (1 + np.exp (-x))

def sigmoid_deriv(x):
 return sigmoid(x) *(1-sigmoid(x))


def forward(x, w1, w2, predict=False):

 a1 = np.matmul(x, w1)

 z1 = sigmoid(a1)

 bias = np.ones ( (len (z1), 1))
 z1 = np. concatenate( (bias, z1), axis=1)
 a2 = np.matmul (z1, w2)
 z2 = sigmoid(a2)
 if predict:
  return z2
 return a1, z1, a2, z2

def backprop(a1, z0, z1, z2, y,w2) :

 delta2 = z2 - y
 Delta2 = np.matmul (z1.T, delta2)
 deltal = (delta2.dot (w2[1: , : ].T)) *sigmoid_deriv (a1)
 Deltal = np.matmul(z0.T, deltal)
 return delta2, Deltal, Delta2

def XOR(inp):
    plt.clf()
    X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    y = np.array([[0], [1], [1], [0]])
    costs = []
    w1 = np.random.randn(3, 5)
    w2 = np.random.randn(6, 1)
    print(w1)

    learningRate = 0.09
    epochs = 15000

    m = len(X)
    for i in range(epochs):

      a1, z1, a2, z2 = forward(X, w1, w2)

      delta2, Deltal, Delta2 = backprop(a1, X, z1, z2, y,w2)

      w1 -= learningRate * (1 / m) * Deltal
      w2 -= learningRate * (1 / m) * Delta2


      c = np.mean(np.abs(delta2))
      costs.append(c)
      if i % 1000 == 0:
        print(f"Iteration: {i}. Error: {c}")
    print("Training complete.")

    # Make predictions
    z3 = forward(X, w1, w2, True)
    print("Percentages:  ")
    print(z3)
    print("Predictions: ")
    print(np.round(z3))



    plt.plot()
    plt.plot(costs,'k-')
    plt.show()

def XNOR(inp):
        plt.clf()
        X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

        y = np.array([[1], [0], [0], [1]])
        costs = []
        w1 = np.random.randn(3, 5)
        w2 = np.random.randn(6, 1)
        print(w1)

        learningRate = 0.09
        epochs = 15000

        m = len(X)
        for i in range(epochs):

            a1, z1, a2, z2 = forward(X, w1, w2)

            delta2, Deltal, Delta2 = backprop(a1, X, z1, z2, y, w2)

            w1 -= learningRate * (1 / m) * Deltal
            w2 -= learningRate * (1 / m) * Delta2

            c = np.mean(np.abs(delta2))
            costs.append(c)
            if i % 1000 == 0:
                print(f"Iteration: {i}. Error: {c}")
        print("Training complete.")

        # Make predictions
        z3 = forward(X, w1, w2, True)
        print("Percentages:  ")
        print(z3)
        print("Predictions: ")
        print(np.round(z3))

        plt.plot()
        plt.plot(costs, 'k-')
        plt.show()
def click_me():
    plt.rc('figure', figsize=(10, 8))
    plt.style.use('fivethirtyeight')
    X = [0, 0, 1, 1]

    Y = [0, 1, 0, 1]
    color = ['red']

    plt.scatter([0, 0, 1], [0, 1, 0], color='blue')
    plt.scatter([1], [1], color='red')

    plt.xlabel('X input feature')
    plt.ylabel('Y input feature')
    plt.title('Perceptron regression for AND Gate')
    plt.plot()
    plt.tight_layout

    if(i.get()==1):
        input = 1
        print("Input : " + str(input) + "Output : " + str(NOTPerceptron(input)))
    elif(i.get()==2):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(AndPerceptron(input)))
    elif(i.get()==3):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(NANDPerceptron(input)))
    elif (i.get() == 4):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(ORPerceptron(input)))
    elif (i.get() == 5):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(NORPerceptron(input)))
    elif (i.get()==6):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(XOR(input)))
    elif (i.get() == 7):
        input = [1, 1]
        print("Input : " + str(input) + "Output : " + str(XNOR(input)))
r1=Radiobutton(root,text="NOT", value=1,variable=i,font="GRIFON 16",bg ="#14003d",fg="#0098FF")
r1.place(x=15,y=160)
r2=Radiobutton(root,text="AND", value=2,variable=i,font="GRIFON 16",bg ="#19013f",fg="#0098FF")
r2.place(x=15,y=210)
r3=Radiobutton(root,text="NAND", value=3,variable=i,font="GRIFON 16",bg ="#1f0141",fg="#0098FF")
r3.place(x=15,y=260)
r4=Radiobutton(root,text="OR", value=4,variable=i,font="GRIFON 16",bg ="#250043",fg="#0098FF")
r4.place(x=15,y=310)
r5=Radiobutton(root,text="NOR",value=5,variable=i,font="GRIFON 16",bg ="#2a0044",fg="#0098FF")
r5.place(x=15,y=360)
r6=Radiobutton(root,text="XOR", value=6,variable=i,font="GRIFON 16",bg ="#2f0046",fg="#0098FF")
r6.place(x=15,y=410)
r7=Radiobutton(root,text="XNOR", value=7,variable=i,font="GRIFON 16",bg ="#350048",fg="#0098FF")
r7.place(x=15,y=460)
button=Button(root,text="START",font=Font_tuple,bg="#FF9700",borderwidth=-3,fg="#ffffff", command=click_me)
button.place(x=223,y=530)

root.mainloop()