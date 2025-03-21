import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageTk
import tkinter as tk

def compress_image(factor, file_path_s):
    if not isinstance(file_path_s, list):
        with Image.open(file_path_s) as img:
            new_size = (img.width//factor, img.height//factor)
            img = img.resize(new_size)
            image_array = np.array(img)
        return image_array, new_size
    else:
        image_arrays = []
        for file_path in file_path_s:
            with Image.open(file_path) as img:
                new_size = (img.width//factor, img.height//factor)
                img = img.resize(new_size)
                image_arrays.append(np.array(img))
        return image_arrays, new_size

def plot_indices(data, sat_n, size, comp, dpi, save_image, res):
    indices = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
    for i, water_index in enumerate(data):
        plt.figure(figsize=(size))
        if sat_n != 2:
            sat_name = "Landsat"
            sat_letter = "L"
        else:
            sat_name = "Sentinel"
            sat_letter = "S"
        plt.title(f"{sat_name} {sat_n} {indices[i]} C{comp} DPI{dpi} R{res}", 
                  fontsize=8)
        
        ax = plt.gca()
        plt.imshow(water_index)
        
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(left=False, bottom=False, 
                       labelleft=False, labelbottom=False)
        
        if save_image:
            print(f"saving {indices[i]} image", end="... ")
            plot_name = f"{sat_letter}{sat_n}_{indices[i]}_C{comp}_DPI{dpi}_R{res}.png"
            
            # check for file name already existing and increment file name
            base_name, extension = os.path.splitext(plot_name)
            counter = 1
            while os.path.exists(plot_name):
                plot_name = f"{base_name}_{counter}{extension}"
                counter += 1
            
            plt.savefig(plot_name, dpi=dpi, bbox_inches="tight")
            print(f"complete! saved as {plot_name}")
        
        print(f"displaying {indices[i]} image", end="... ")
        plt.show()
        print(f"{indices[i]} image display complete!")

def upscale_image_array(img_array, factor=2):
    return np.repeat(np.repeat(img_array, factor, axis=0), factor, axis=1)

def get_rgb(blue_path, green_path, red_path, save_image, res, show_image):
    bands = []
    print("creating RGB image", end="... ")
    for path in (blue_path, green_path, red_path):
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)
            bands.append(((arr / arr.max()) * 255).astype(np.uint8))
    rgb_array = np.stack(bands, axis=-1)
    rgb_image = Image.fromarray(rgb_array)
    print("complete!")
    if save_image:
        print("saving image", end="... ")
        rgb_image.save(f"{res}m_RGB.png")
        print("complete!")
    if show_image:
        print("displaying image", end="... ")
        rgb_image.show()
        print("complete!")
    return rgb_array

def mask_sentinel(path, high_res, image_arrays, comp):
    if high_res:
        image_arrays[-1] = upscale_image_array(image_arrays[-1], factor=2)
        image_arrays[-2] = upscale_image_array(image_arrays[-2], factor=2)
        path = path + "MSK_CLDPRB_20m.jp2"
        clouds_array, size = compress_image(comp, path)
        clouds_array = upscale_image_array(clouds_array, factor=2)
    else:
        path = path + "MSK_CLDPRB_60m.jp2"
        clouds_array, size = compress_image(comp, path)
    
    clouds_array = np.where(clouds_array > 50, 100, clouds_array)
    cloud_positions = np.argwhere(clouds_array == 100)
    
    for image_array in image_arrays:
        image_array[cloud_positions[:, 0], cloud_positions[:, 1]] = 0
    
    return image_arrays

def find_rgb_file(path):
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path): # if item is a folder
            found_rgb, rgb_path = find_rgb_file(full_path)
            if found_rgb:  
                return True, rgb_path
        else: # if item is a file
            if "RGB" in item and "10m" in item and "bright" in item:
                return True, full_path
    return False, None

def prompt_roi(image_array: np.ndarray):
    """
    Opens a Tkinter window displaying the image (as a numpy array).
    Allows the user to select multiple ROIs by click-and-drag.
    After each selection, click "Save ROI" to confirm that region.
    When done, click "Finish" to close the window and return a list of ROI coordinates.
    
    Returns:
        A list of tuples (x, y, width, height) for each ROI.
    """
    # Convert the numpy array to a PIL image
    image = Image.fromarray(image_array)
    image = image.resize((500, 500))
    width, height = image.size

    rois = []          # List to store confirmed ROI coordinates
    current_roi = None # To temporarily hold the current ROI coordinates
    start_x = start_y = 0
    current_rect = None

    # Create the Tkinter window and canvas
    root = tk.Tk()
    root.title("Select Multiple ROIs")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()

    # Display the image on the canvas
    tk_image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    # Event handlers for drawing the ROI rectangle
    def on_button_press(event):
        nonlocal start_x, start_y, current_rect, current_roi
        start_x, start_y = event.x, event.y
        # Start drawing a rectangle
        current_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, 
                                               outline="red", width=2)
        current_roi = None  # Reset current ROI

    def on_move_press(event):
        nonlocal current_rect
        # Update the rectangle as the mouse is dragged
        canvas.coords(current_rect, start_x, start_y, event.x, event.y)

    def on_button_release(event):
        nonlocal current_roi
        # Finalize the ROI on mouse release
        end_x, end_y = event.x, event.y
        x = min(start_x, end_x)
        y = min(start_y, end_y)
        w = abs(end_x - start_x)
        h = abs(end_y - start_y)
        current_roi = (x, y, w, h)
        # The rectangle remains on screen until confirmed

    # Button callback to save the current ROI
    def save_roi():
        nonlocal rois, current_roi, current_rect
        if current_roi is not None:
            rois.append(current_roi)
            # Optionally, change the rectangle color to indicate it was saved
            canvas.itemconfig(current_rect, outline="green")
            # Reset current selection variables for the next ROI
            current_roi = None
            # Prepare for a new ROI by not keeping the reference to this rectangle
            current_rect = None

    # Button callback to finish the ROI selection and close the window
    def finish_selection():
        root.destroy()

    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    # Add "Save ROI" and "Finish" buttons
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, pady=10)

    save_button = tk.Button(button_frame, text="Save ROI", command=save_roi)
    save_button.pack(side=tk.LEFT, padx=10)

    finish_button = tk.Button(button_frame, text="Finish", command=finish_selection)
    finish_button.pack(side=tk.RIGHT, padx=10)

    root.mainloop()
    return rois
