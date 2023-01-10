import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Radio Button Example")
def button_clicked():
    print(radio_var.get())

# Create the radio buttons
radio_var = tk.StringVar()
radio_button1 = tk.Radiobutton(window, text="Option 1", variable=radio_var, value="option1")
radio_button2 = tk.Radiobutton(window, text="Option 2", variable=radio_var, value="option2")
radio_button3 = tk.Radiobutton(window, text="Option 3", variable=radio_var, value="option3")
radio_button4 = tk.Radiobutton(window, text="Option 4", variable=radio_var, value="option4")
radio_button5 = tk.Radiobutton(window, text="Option 5", variable=radio_var, value="option5")
radio_button6 = tk.Radiobutton(window, text="Option 6", variable=radio_var, value="option6")
radio_button7 = tk.Radiobutton(window, text="Option 7", variable=radio_var, value="option7")

# Create the button
button = tk.Button(window, text="Submit", command=button_clicked)

# Pack the radio buttons and button into the window
radio_button1.pack()
radio_button2.pack()
radio_button3.pack()
radio_button4.pack()
radio_button5.pack()
radio_button6.pack()
radio_button7.pack()
button.pack()

# Run the main loop
window.mainloop()
