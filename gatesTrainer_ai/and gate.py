
def UnitStep(inp):
    if inp >= 0:
        return 1
    else:
        return 0
def perceptron(and_input):

  a = [0,0, 1,1]

  b = [0,1,0,1]
  y = [0,0,0,1] # Actual Output

  w = [0.3,-0.1]

  threshold = 0.2

  learning_rate = 0.1
  i=0
  while i<16 :
      Yd = (a[i%4]*w[0]+b[i%4]*w[1])-threshold
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
  Yd = (and_input[0] * w[0] + and_input[1] * w[1]) - threshold
  return UnitStep(Yd)

and_input = [0,1]
print("Input : " + str(and_input) + "Output : " + str(perceptron(and_input)) )
