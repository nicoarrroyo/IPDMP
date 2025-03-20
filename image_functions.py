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

def prompt_roi(PATH):
    """
    region of interest code using tkinter
    https://stackoverflow.com/questions/55636313/selecting-an-area-of-an-image-wi
    th-a-mouse-and-recording-the-dimensions-of-the-s
    question asked by InfiniteLoop on April 11 2019
    question answered by martineau on April 11 2019
    """
    class MousePositionTracker(tk.Frame):
        """Tkinter Canvas mouse position widget."""
        def __init__(self, canvas):
            self.canvas = canvas
            self.canv_width = self.canvas.cget('width')
            self.canv_height = self.canvas.cget('height')
            self.reset()
    
            # Create canvas cross-hair lines.
            xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
            self.lines = (
                self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts)
            )
    
        def cur_selection(self):
            return (self.start, self.end)
    
        def begin(self, event):
            self.hide()
            self.start = (event.x, event.y)  # Remember starting position
    
        def update(self, event):
            self.end = (event.x, event.y)
            self._update(event)
            self._command(self.start, (event.x, event.y))  # User callback
    
        def _update(self, event):
            # Update cross-hair lines.
            self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
            self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
            self.show()
    
        def reset(self):
            self.start = self.end = None
    
        def hide(self):
            self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
            self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)
    
        def show(self):
            self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
            self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)
    
        def autodraw(self, command=lambda *args: None):
            """Setup automatic drawing; supports a command callback."""
            self.reset()
            self._command = command
            self.canvas.bind("<Button-1>", self.begin)
            self.canvas.bind("<B1-Motion>", self.update)
            self.canvas.bind("<ButtonRelease-1>", self.quit)
    
        def quit(self, event):
            self.hide()  # Hide cross-hairs.
            self.reset()  # This resets the selection coordinates
    
    
    class SelectionObject:
        """Widget to display a rectangular area on a 
        given canvas defined by two points."""
        def __init__(self, canvas, select_opts):
            self.canvas = canvas
            self.select_opts1 = select_opts
            self.width = self.canvas.cget('width')
            self.height = self.canvas.cget('height')
    
            # Options for areas outside the rectangular selection.
            select_opts1 = self.select_opts1.copy()
            select_opts1.update(state=tk.HIDDEN)  # Hide initially.
            # Separate options for area inside the rectangular selection.
            select_opts2 = dict(dash=(2, 2), fill='', outline='white', 
                                state=tk.HIDDEN)
    
            # Initial extrema for inner and outer rectangles.
            imin_x, imin_y, imax_x, imax_y = 0, 0, 1, 1
            omin_x, omin_y, omax_x, omax_y = 0, 0, self.width, self.height
    
            self.rects = (
                # Areas *outside* the selection rectangle.
                self.canvas.create_rectangle(omin_x, omin_y, omax_x, imin_y, 
                                             **select_opts1),
                self.canvas.create_rectangle(omin_x, imin_y, imin_x, imax_y, 
                                             **select_opts1),
                self.canvas.create_rectangle(imax_x, imin_y, omax_x, imax_y, 
                                             **select_opts1),
                self.canvas.create_rectangle(omin_x, imax_y, omax_x, omax_y, 
                                             **select_opts1),
                # The inner rectangle (selected area).
                self.canvas.create_rectangle(imin_x, imin_y, imax_x, imax_y, 
                                             **select_opts2)
            )
    
        def update(self, start, end):
            imin_x, imin_y, imax_x, imax_y = self._get_coords(start, end)
            omin_x, omin_y, omax_x, omax_y = 0, 0, self.width, self.height
    
            # Update coordinates for all rectangles.
            self.canvas.coords(self.rects[0], omin_x, omin_y, omax_x, imin_y)
            self.canvas.coords(self.rects[1], omin_x, imin_y, imin_x, imax_y)
            self.canvas.coords(self.rects[2], imax_x, imin_y, omax_x, imax_y)
            self.canvas.coords(self.rects[3], omin_x, imax_y, omax_x, omax_y)
            self.canvas.coords(self.rects[4], imin_x, imin_y, imax_x, imax_y)
    
            for rect in self.rects:
                self.canvas.itemconfigure(rect, state=tk.NORMAL)
    
        def _get_coords(self, start, end):
            return (min(start[0], end[0]), min(start[1], end[1]),
                    max(start[0], end[0]), max(start[1], end[1]))
    
        def hide(self):
            for rect in self.rects:
                self.canvas.itemconfigure(rect, state=tk.HIDDEN)
    
    
    class Application(tk.Frame):
        # Default selection options.
        SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red', outline='')
    
        def __init__(self, parent, *args, **kwargs):
            super().__init__(parent, *args, **kwargs)
    
            self.path = PATH
            self.img = Image.open(self.path)
            self.tk_img = ImageTk.PhotoImage(self.img)
    
            self.canvas = tk.Canvas(root, width=self.tk_img.width(), 
                                    height=self.tk_img.height(),
                                    borderwidth=0, highlightthickness=0)
            self.canvas.pack(expand=True)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
            self.canvas.img = self.tk_img  # Keep a reference.
    
            # Create a selection object to show current selection boundaries.
            self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
    
            # Callback function to update selection boundaries.
            def on_drag(start, end, **kwargs):
                self.selection_obj.update(start, end)
    
            # Create mouse position tracker and enable callbacks.
            self.posn_tracker = MousePositionTracker(self.canvas)
            self.posn_tracker.autodraw(command=on_drag)
    
            # Add a button to save the selected region.
            self.save_button = tk.Button(self, text="Save & Quit", 
                                         command=self.save_selection)
            self.save_button.pack()
    
            # This attribute will store the final selection coordinates.
            self.final_selection = None
    
            # Override the default ButtonRelease binding to store the
            # final selection before it's reset.
            original_quit = self.posn_tracker.quit
    
            def new_quit(event):
                # Capture the current selection before resetting.
                self.final_selection = self.posn_tracker.cur_selection()
                original_quit(event)
            self.posn_tracker.canvas.bind("<ButtonRelease-1>", new_quit)
    
        def save_selection(self):
            """Crop and save the selected region as a new image."""
            selection = self.final_selection
            # print(selection)
            # globals()["selec"] = selection
            
            if selection and selection[0] and selection[1]:
                start, end = selection
                min_x, min_y, max_x, max_y = self.selection_obj._get_coords(start, end)
                cropped_image = self.img.crop((min_x, min_y, max_x, max_y))
                cropped_image.save("selected_area.png")
                print("Selected area saved as 'selected_area.png'.")
                root.destroy()
            else:
                print("No area selected")
    
    if __name__ == '__main__':
        WIDTH, HEIGHT = 900, 600
        BACKGROUND = 'grey'
        TITLE = 'Image Cropper'
        
        root = tk.Tk()
        root.title(TITLE)
        root.geometry(f'{WIDTH}x{HEIGHT}')
        root.configure(background=BACKGROUND)
        
        app = Application(root, background=BACKGROUND)
        app.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
        root.mainloop()
