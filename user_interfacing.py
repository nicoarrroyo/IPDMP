import sys
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk

def table_print(**kwargs):
    """
    The name of the variable and the value is outputted together. 
    An orderly way of outputting variables at the start of a program. 
    
    Parameters
    ----------
    **kwargs : any
        Any number of any type of variable is passed and outputted. 
    
    Returns
    -------
    None.
    
    """
    if not kwargs:
        print("No data to display.")
        return
    
    # Compute max lengths
    max_var_length = max(map(len, kwargs.keys()), default=8)
    max_value_length = max(map(lambda v: len(str(v)), kwargs.values()), default=5)
    
    # Format the header and separator dynamically
    header = f"| {'Variable'.ljust(max_var_length)} | {'Value'.ljust(max_value_length)} |"
    separator = "-" * len(header)
    
    # Print table
    print(separator)
    print(header)
    print(separator)
    for key, value in kwargs.items():
        print(f"| {key.ljust(max_var_length)} | {str(value).ljust(max_value_length)} |")
    print(separator)

def spinner(stop_event, message):
    """
    A simple spinner that runs until stop_event is set.
    The spinner updates the ellipsis by overwriting the same line.
    """
    chase = ["   ", ".  ", ".. ", "...", " ..", "  .", "   ", 
             "  .", " ..", "...", ".. ", ".  "]
    wobble = [" \ ", " | ", " / ", " | "]
    woosh = ["|   ", ")   ", " )  ", "  ) ", "   )", "   |", 
             "   |", "  ( ", " (  ", "(   ", "|   "]
    ellipses = ["   ", ".  ", ".. ", "..."]
    dude = [" :D ", " :) ", " :\ ", " :( ", " :\ ", " :) "]
    
    frames = chase
    frames = wobble
    frames = woosh
    frames = ellipses
    frames = dude
    
    frames = wobble
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write("\r" + message + frame)
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    # Clear the spinner message on stop
    sys.stdout.write("\r" + message + "... complete! \n")
    sys.stdout.flush()

def start_spinner(message):
    # Create an event for signaling the spinner to stop
    stop_event = threading.Event()
    
    # Use a thread to run the spinner concurrently
    thread = threading.Thread(target=spinner, args=(stop_event, message))
    thread.start()
    return stop_event, thread

def end_spinner(stop_event, thread):
    # Turn off the spinner once processing is done
    stop_event.set()
    thread.join()

def prompt_roi(image_array, n):
    """
    Opens a Tkinter window displaying the image (as a numpy array).
    Allows the user to select multiple ROIs by click-and-drag.
    The ROI is automatically saved upon mouse release.
    When done, click the final button to close the window and return a 
    list of ROI coordinates.
    
    Parameters
    ----------
    image_array : numpy array
        A numpy array converted from an image. 
    n : int
        An integer representing the number regions of interest (ROIs) that 
        were identified by the user. 
    
    Returns
    -------
    rois : list
        List of integers (upper-left x-coordinate (ulx), 
                          upper-left y-coordinate (uly), 
                          bottom-right x-coordinate (brx), 
                          bottom-right y-coordinate (bry)) for each ROI. 
    
    """
    # Convert the numpy array to a PIL image
    image = Image.fromarray(image_array)
    image = image.resize((500, 500))  # Resize the image to fit in the window
    width, height = image.size
    
    rois = []          # List to store confirmed ROI coordinates
    rects = []         # List to store the drawn rectangles
    current_roi = None # To temporarily hold the current ROI coordinates
    current_rect = None
    start_x = start_y = 0
    
    error_counter = 0
    while error_counter < 2:
        try:
            # Create the Tkinter window and canvas
            root = tk.Tk()
            root.title("Select Regions of Interest (ROIs)")
            root.resizable(False, False)
            canvas = tk.Canvas(root, width=width, height=height)
            canvas.pack()
    
            # Display the image on the canvas
            tk_image = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor="nw", image=tk_image)
            break
        except:
            error_counter += 1
            root = tk.Toplevel()
            root.title("CLOSE THIS WINDOW")
            canvas = tk.Canvas(root, width=width, height=height)
            canvas.pack()
            root.destroy()
            print("Please close any windows that were opened")
            root.mainloop()
    if error_counter >= 2:
        print("Broken prompt_roi function")
        return
    
    # Create the lines for following the cursor
    vertical_line = canvas.create_line(0, 0, 0, height, fill="red", dash=(4, 2))
    horizontal_line = canvas.create_line(0, 0, width, 0, fill="red", dash=(4, 2))
    
    # Helper function to update the status bar message.
    def set_status(msg):
        status_label.config(text=msg)
    
    # Event handlers for drawing the ROI rectangle
    def on_button_press(event):
        nonlocal start_x, start_y, current_rect, current_roi
        start_x, start_y = event.x, event.y
        if current_rect is not None:
            canvas.delete(current_rect)  # Delete the previous rectangle if exists
        # Start drawing a rectangle
        current_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, 
                                               outline="red", width=2)
        current_roi = None  # Reset current ROI
    
    def on_move_press(event):
        nonlocal current_rect
        # Update the rectangle as the mouse is dragged
        canvas.coords(current_rect, start_x, start_y, event.x, event.y)
        # Update the follower lines by resetting their coordinates
        canvas.coords(vertical_line, event.x, 0, event.x, height)
        canvas.coords(horizontal_line, 0, event.y, width, event.y)
    
    def on_button_release(event):
        nonlocal current_roi
        # Finalize the ROI on mouse release
        end_x, end_y = event.x, event.y
        ulx = min(start_x, end_x)
        uly = min(start_y, end_y)
        brx = max(end_x, start_x)
        bry = max(end_y, start_y)
        current_roi = (ulx, uly, brx, bry)
        # Auto-save the ROI upon release
        save_roi()
    
    # Event handler for mouse motion (unpressed button) to update the lines
    def on_mouse_motion(event):
        canvas.coords(vertical_line, event.x, 0, event.x, height)
        canvas.coords(horizontal_line, 0, event.y, width, event.y)
    
    # Function to save the current ROI automatically
    def save_roi():
        nonlocal rois, current_roi, rects, current_rect
        if len(rois) < n:
            if current_roi is not None:
                # Ensure the ROI coordinates are within bounds
                current_roi = [max(0, int(roi)) for roi in current_roi]
                current_roi = [min(width, int(roi)) for roi in current_roi]
                
                rois.append(current_roi)
                rects.append(current_rect)  # Keep track of the rectangle reference
                canvas.itemconfig(current_rect, outline="green")
                set_status((f"Saved ROI {current_roi}. {n-len(rois)} left"))
                # Reset current selection variables for the next ROI
                current_roi = None
                current_rect = None
            else:
                set_status("No region of interest selected")
        else:
            set_status(f"Too many selections, expected: {n}. Overwrite a selection.")
    
    # Button callback to finish the ROI selection and close the window
    def finish():
        nonlocal rois
        # If there's a ROI in progress, try to save it.
        if current_roi is not None:
            save_roi()
        if len(rois) < n:
            set_status(f"{n - len(rois)} selection(s) remaining")
        else:
            root.destroy()
    
    def overwrite():
        nonlocal rois, current_roi, current_rect, rects
        # Remove the current drawing if any.
        canvas.delete(current_rect)
        if rois and rects:
            # Remove the last saved ROI and rectangle
            canvas.delete(rects[-1])
            rois.pop()  # Remove the last ROI coordinates
            rects.pop()  # Remove the last rectangle reference
            current_rect = None  # Reset the current rectangle reference
            current_roi = None  # Reset the current ROI coordinates
            set_status("Overwritten ROI")
        else:
            set_status("No regions of interest saved")
    
    def hide_cursor(event):
        canvas.config(cursor="none")
    
    def show_cursor(event):
        canvas.config(cursor="")
    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    canvas.bind("<Motion>", on_mouse_motion)
    canvas.config(cursor="none")
    canvas.bind("<Enter>", lambda event: canvas.config(cursor="none"))
    canvas.bind("<Leave>", lambda event: canvas.config(cursor=""))
    
    # Create button frame and add "Overwrite" and "Finish" buttons only.
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, pady=10)
    
    overwrite_button = tk.Button(button_frame, text="Overwrite", command=overwrite)
    overwrite_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    # Set initial text based on expected number of ROIs.
    finish_button = tk.Button(button_frame, text="Finish", command=finish)
    finish_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    # Create the status bar below the buttons
    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, padx=2, pady=2)
    
    root.mainloop()
    return rois