import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

def compress_images(factor, width, height, images):
    image_arrays = []
    if factor != 1:
        new_size = (width//factor, height//factor)
    else:
        new_size = (width, height)
    for img in images:
        img = img.resize(new_size)
        image_arrays.append(np.array(img))
    return images, image_arrays, new_size

def plot_image(data, landsat_n, size,  minimum, maximum, comp, dpi, save_image):
    indices = ['NDWI', 'MNDWI', 'AWEI Shadowed', 'AWEI Non-Shadowed']
    for i, water_index in enumerate(data):
        plt.figure(figsize=(size))
        plt.title(f'Landsat {landsat_n} {indices[i]} C{comp} DPI{dpi}', fontsize=8)
        
        ax = plt.gca() # get current axis
        im = plt.imshow(water_index, 
                        interpolation='nearest', cmap='viridis', 
                        vmin=minimum, vmax=maximum)
        axins = inset_axes(ax, width='10%', height='2%', loc='lower right')
        
        cbar = plt.colorbar(im, cax=axins, orientation='horizontal')
        cbar.set_ticks(np.linspace(minimum, maximum, 3))
        cbar.set_ticklabels([f'{minimum}', 
                             f'{int(np.average([minimum, maximum]))}', 
                             f'{maximum}'], 
                            fontsize=5, color='w')
        axins.xaxis.set_ticks_position('top')
        
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False, 
                       labelleft=False, labelbottom=False)
        
        if save_image:
            print(f'saving {indices[i]} image', end='... ')
            plot_name = f'L{landsat_n}_{indices[i]}_C{comp}_DPI{dpi}_SIZE{size}.png'
            
            # check for file name already existing and increment file name
            base_name, extension = os.path.splitext(plot_name)
            counter = 1
            while os.path.exists(plot_name):
                plot_name = f'{base_name}_{counter}{extension}'
                counter += 1
            
            plt.savefig(plot_name, dpi=dpi, bbox_inches='tight')
            print(f'complete! saved as {plot_name}')
        
        print(f'displaying {indices[i]} image', end='... ')
        plt.show()

def rotate_image(show_images, im):
    if show_images:
        black = {}
        print('opening image')
        for rotation_angle in range(0, 3):
            print(f'Rotation: {rotation_angle}')
            black_pixel_count = 0
            img = im.rotate(rotation_angle, expand=0)
            img_array = np.array(img)
            for pixel_row in img_array:
                for pixel in pixel_row:
                    if pixel == 0:
                        black_pixel_count += 1
            black[rotation_angle] = black_pixel_count
            
        im.show()
        print('image open')
        print(black)
