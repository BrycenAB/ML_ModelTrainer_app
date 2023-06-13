import tkinter as tk
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import Controller

# main window
window = tk.Tk()
window.title("ML Model Trainer")
window.configure(width=1100, height=800)
# set the minimum window size
window.minsize(1100, 800)

# change background color of window to Black
window.configure(background='black')

# tk variables for the tkinter entry widgets
Test_data_size = tk.StringVar(window)
Training_total = tk.StringVar(window)
Random_state = tk.StringVar(window)

selected_datapoint = "None"


def init_tkVars():
    # set the default values for the tkinter variables
    Test_data_size.set("0.2")
    Training_total.set("30")
    Random_state.set("42")


# Button to upload a CSV file
upload_button = tk.Button(text="Upload .csv file", command=lambda: Controller.upload_csv(),
                          bg="black", fg="green", activebackground="green", activeforeground="black")
upload_button.pack()

# reset all button
reset_button = tk.Button(text="Reset All", command=lambda: Controller.reset_all(), bg="red", fg="black",
                         activebackground="green", activeforeground="black")
reset_button.pack(side="bottom")

# Labels and input for test/training split
data_size_label = tk.Label(text="Test Data Size:", bg="black", fg="green")
data_size_label.place(x=10, y=0)
data_size_entry = tk.Entry(window, width=10, textvariable=Test_data_size)
data_size_entry.place(x=10, y=20)
training_total_label = tk.Label(text="Training Rounds:", bg="black", fg="green")
training_total_label.place(x=10, y=40)
training_total_entry = tk.Entry(window, width=10, textvariable=Training_total)
training_total_entry.place(x=10, y=60)
# label and entry for random_state
random_state_label = tk.Label(text="Random State:", bg="black", fg="green")
random_state_label.place(x=10, y=80)
random_state_entry = tk.Entry(window, width=10, textvariable=Random_state)
random_state_entry.place(x=10, y=100)

# button to run test/train split

test_train_button = tk.Button(text="Test/Train Split",
                              command=lambda: Controller.train_and_test_model(selected_datapoint,
                                                                              float(data_size_entry.get()),
                                                                              int(random_state_entry.get()),
                                                                              int(training_total_entry.get())),
                              bg="black", fg="green", activebackground="green", activeforeground="black")
test_train_button.place(x=10, y=120)

# create a label and text box to display the accuracy of the model
accuracy_label = tk.Label(text="Accuracy:", bg="black", fg="green")
accuracy_label.place(x=10, y=150)
accuracy_text = tk.Text(window, height=1, width=10)
accuracy_text.insert(tk.END, '0.0')
accuracy_text.place(x=10, y=170)

# text box to display name of csv file
csv_file_text = tk.Text(window, height=1, width=30)
csv_file_text.insert(tk.END, "No file selected")
csv_file_text.configure(state="disabled")
csv_file_text.place(x=10, y=200)

# button to rescale data
rescale_button = tk.Button(text="Rescale Data", command=lambda: Controller.rescale_data(),
                           bg="black", fg="green", activebackground="green", activeforeground="black")
rescale_button.place(x=260, y=170)
# button to standardize data
standardize_button = tk.Button(text="Standardize Data", command=lambda: Controller.standardize_data(),
                               bg="black", fg="green", activebackground="green", activeforeground="black")
standardize_button.place(x=338, y=170)

# button to normalize data
normalize_button = tk.Button(text="Normalize Data", command=lambda: Controller.normalize_data(),
                             bg="black", fg="green", activebackground="green", activeforeground="black")
normalize_button.place(x=260, y=200)

# button to binarize data
binarize_button = tk.Button(text="Binarize Data", command=lambda: Controller.binarize_data(),
                            bg="black", fg="green", activebackground="green", activeforeground="black")
binarize_button.place(x=353, y=200)

# create an entry to name the pickle file
pickle_name_label = tk.Label(text="Pickle File Name:", bg="black", fg="green")
pickle_name_label.place(x=120, y=0)
pickle_name_entry = tk.Entry(window, width=20)
pickle_name_entry.place(x=120, y=20)
# button to save the model as a pickle file
save_model_button = tk.Button(text="Save Model", command=lambda: Controller.save_model(pickle_name_entry.get()),
                              bg="black", fg="green", activebackground="green", activeforeground="black")
save_model_button.place(x=120, y=40)

# button to upload pickle model
upload_pickle_button = tk.Button(text="Upload Pickle Model", command=lambda: Controller.upload_pickle_model(),
                                 bg="black", fg="green", activebackground="green", activeforeground="black")
# place the upload pickle button on the far right of the window and make it move with the window size
upload_pickle_button.place(relx=1.0, rely=0.0, anchor='ne')

# create text box to show name of file just uploaded with upload pickle button
pickle_file_name = tk.Text(window, height=1, width=30)
pickle_file_name.insert(tk.END, "No file selected")
# disable pickle file name text box
pickle_file_name.configure(state="disabled")
pickle_file_name.place(relx=1.0, rely=0.039, anchor='ne')

# input for user values to predict
input_text = tk.Entry(window, width=40)
input_text.place(relx=1.0, rely=0.07, anchor='ne')

# button to run the model
run_model_button = tk.Button(text="Run Model", command=lambda: Controller.make_prediction(
    'pickle_models/curr_best_model.pickle', input_text.get(), selected_datapoint),
                             bg="black", fg="green",
                             activebackground="green", activeforeground="black")
run_model_button.place(relx=1.0, rely=0.1, anchor='ne')


# function to return the selected primary datapoint                       <----- Do i need to move this to controller???
def select_datapoint(event):
    global selected_datapoint
    selected_datapoint = datapoint_dropdown.get(datapoint_dropdown.curselection())
    return selected_datapoint

# function to create a matplotlib graph on a popup window
def create_graph():
    # create a new window
    graph_window = tk.Toplevel(window)
    graph_window.title("Graph")
    graph_window.geometry("800x600")
    graph_window.configure(bg="black")
    # create a canvas to display the graph
    fig = Figure(figsize=(8, 6), dpi=100)
    graph_canvas = FigureCanvasTkAgg(fig, master=graph_window)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # create a toolbar to interact with the graph
    toolbar = NavigationToolbar2Tk(graph_canvas, graph_window)
    toolbar.update()
    graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# button to show graph
#graph_button = tk.Button(text="Show Graph", command=lambda: create_graph(),
#                            bg="black", fg="green", activebackground="green", activeforeground="black")
#graph_button.place(relx=1.0, rely=0.13, anchor='ne')


# Create a dropdown to select the primary datapoint
datapoint_label = tk.Label(text="Select datapoint to predict:", bg="black", fg="green")
datapoint_label.pack()
datapoint_dropdown = tk.Listbox(bg="black", fg="green", activestyle="none", selectbackground="green", )
datapoint_dropdown.pack()

# Bind the dropdown to the "select_datapoint" function
datapoint_dropdown.bind("<<ListboxSelect>>", select_datapoint)

# label and output window
output_label = tk.Label(text="CSV Data:", bg="black", fg="green")
output_label.pack()
output_text = tk.Text(bg='black', fg='green', cursor='arrow', width=90, height=30)
# make output text box fill the right half of the bottom half of the window
output_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
output_scrollbar = tk.Scrollbar(command=output_text.yview)
output_text.configure(yscrollcommand=output_scrollbar.set)
output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# label for output window2
log_label = tk.Label(text="Log:", bg="black", fg="green")
log_label.pack()
# output window2
log_text = tk.Text(bg='black', fg='green', cursor='arrow', width=40, height=30, font=("Helvetica", 8))
# make output text box fill the left half of the bottom half of the window
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
output_scrollbar2 = tk.Scrollbar(command=log_text.yview)
log_text.configure(yscrollcommand=output_scrollbar2.set)
output_scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

# call a function when window is closed
window.protocol("WM_DELETE_WINDOW", Controller.reset_all())

init_tkVars()
