from ex5_helper import *
from typing import Optional
import math


def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    """
    The function get image in rows x columns x channels x and return
    list of images that each of them represent color channel
    :param image:3D list
    :return:3D list
    """
    result = []
    # run for each row
    for i in range(len(image[0][0])):
        channel = []
        # run for each column(j lists that each list is a column)
        for j in range(len(image)):
            row_in_channel = []
            # run for each pixle
            for k in range(len(image[0])):
                row_in_channel.append(image[j][k][i])
            channel.append(row_in_channel)
        result.append(channel)
    return result



def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    """
    The function get a channels-long list of two-dimensional images consisting of individual color channels
    and combines them into a single image of dimensions rows x columns x channels
    :param image:3D list
    :return:3D list
    """
    result = []
    # run for each pixle
    for i in range(len(channels[0])):
        row = []
        # run for each column(j lists that each list is a column)
        for j in range(len(channels[0][0])):
            pixel = []
            # run for each row
            for k in range(len(channels)):
                pixel.append(channels[k][i][j])
            row.append(pixel)
        result.append(row)
    return result

def calc_RGB2grayscale(c_red,c_green,c_blue):
    """
    The function performs a certain averaging
    of the values of the colored sculpture into one value
    :param pixel:Three values that represent
    shades of color - red, green and blue. float/float
    :return:float
    """
    pixel = 0.299 * c_red + 0.587 * c_green + 0.114 * c_blue
    return pixel


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """
    The function receives a color image (a three-dimensional list) and returns an image
    in black and white tones (a two-dimensional list).
    :param colored_image:3D list
    :return:A two-dimensional list
    """
    # In this func we don't use three for loops because the pixel will always have three columns according
    # the A condition that maintains input integrity
    gray_ret_image = []
    # run for each row
    for i in range(len(colored_image)):
        row = []
        # run for each column
        for j in range(len(colored_image[0])):
            pixel = calc_RGB2grayscale(colored_image[i][j][0], colored_image[i][j][1], colored_image[i][j][2])
            # check that pixel is not none and we can use round func.
            if pixel != None:
                # Take the nearest whole value
                pixel = round(pixel)
                row.append(pixel)
        gray_ret_image.append(row)
    return gray_ret_image


def blur_kernel(size: int) -> Kernel:
    """
    The function accepts a positive and odd integer.
    Returns a kernel (two-dimensional list) of size x size
    where each cell has one value divided by size squared.
    :param size:int size
    :return:size x size list
    """
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 / size ** 2)
        result.append(row)
    return result

def is_in_board(image,left_in_y,left_in_x):
    """
    The function return if both left places from the column and row is in the row.
    :return:bool variable-True or Falae
    """
    if 0 <= left_in_y and left_in_y < len(image) and 0 <= left_in_x and left_in_x < len(image[0]):
        return True
    return False

def multiply_kernel_by_current(i,j,kernel,image):
    """
    The function calculat the result of kernel(list size x size) by the current place in image.
    :return: float result
    """
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    kernel_middle = kernel_height // 2
    result=0
    for y in range(kernel_height):
        for x in range(kernel_width):
            # count the places around the right pixel
            left_in_y = i + y - kernel_middle
            left_in_x = j + x - kernel_middle
            # check if the place is out of the image
            if is_in_board(image, left_in_y, left_in_x):
                result += image[left_in_y][left_in_x] * kernel[y][x]
            else:
                result += image[i][j] * kernel[y][x]
    return result


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    """
    The function receives an image with a single color channel (i.e. a two-dimensional list) and returns an
    image of the same size as the original image when each pixel in the new image is calculated by running the kernel on it.
    Identify the pixel [column]image[row] with the central entry in the kernel matrix, and sum the values of its neighbors
    (including the pixel itself) times their corresponding entry in the kernel.
    :param image:
    :param kernel:
    :return:
    """
    # calculate how many pixels are next to the middle number in kernel,
    # the number is the amount of the numbers from right to left and below and above.
    ret_image = []
    for i in range(len(image)):
        row = []
        # run for each column in the row and calc the new value for each place.
        for j in range(len(image[0])):
            # call the function that calc the result for multiply the kernel by the current image[i][j]
            sum_of_pixel = multiply_kernel_by_current(i, j, kernel, image)
            if sum_of_pixel < 0:
                sum_of_pixel = 0
            elif sum_of_pixel > 255:
                sum_of_pixel = 255
            row.append(round((sum_of_pixel)))
        ret_image.append(row)
    return ret_image

def four_points_around(pixel):
    """
     Returns the four places around the pixel
    :param pixel:list[]
    :return:list[[]]
    """
    y=pixel[0]
    x = pixel[1]
    if y==0:
        ceil_row=1
    else:
        ceil_row = math.ceil(y)
    if x==0:
        ceil_column=1
    else:
        ceil_column = math.ceil(x)
    #calculate therange
    floor_row=math.floor(y)
    floor_column = math.floor(x)
    return [floor_row,floor_column],[ceil_row,floor_column],[floor_row,ceil_column],[ceil_row,ceil_column]

def value_in_x_y(image,a,b,c,d):
    """
    A function that receives four positions
    in the image and returns their value in the image
    :param image:list in list
    :param a,b,c,d:list, for example [0,2]
    :return:int
    """
    return image[a[0]][a[1]],image[b[0]][b[1]],image[c[0]][c[1]],image[d[0]][d[1]]

def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    """
    A function that receives an image with a single color channel (a two-dimensional list)
    and the coordinate of a pixel from the target image as they "fall" in the source image.
    :param image: source image
    :param y:row
    :param x:column
    :return:The pixel value, an integer.
    """
    pixel = [y, x]
    a, b, c, d = four_points_around(pixel)
    a_val, b_val, c_val, d_val = value_in_x_y(image, a, b, c, d)
    delta_y = y - a[0]
    delta_x = x - a[1]
    new_image = a_val * (1 - delta_x) * (1 - delta_y) + b_val * delta_y * (1 - delta_x) + c_val * delta_x * (
                1 - delta_y) + d_val * delta_x * delta_y
    return int(new_image)

def calc_fall_pixel(new_height,new_width,image,pixel):
    """
    The function get an image(list in list)and a pixel in destination and calc the
    value of x and y of the pixel in the source image.
    :return:pixel list in list[[]]
    """
    image_height=len(image)
    image_width=len(image[0])
    # To get the new pixel we will take the y and x values and divide them by
    # the range of the row and column in the target image.
    #one is subtracted because the range is from 0 to the last index
    new_x=pixel[1]/(new_width-1)
    new_y=pixel[0]/(new_height-1)
    # Multiply the row and column range of the source image
    new_x=new_x*(image_width-1)
    new_y=new_y*(image_height-1)
    return [new_y,new_x]

def value_for_corners(i,j,image,new_width,new_height):
    """
    The function returns the value
    that should be in the current corner
    :return:list[]
    """
    #Top left corner
    if i == 0 and j == 0:
        return image[i][j]
    #Top right corner
    elif i==0 and j==new_width-1:
        return image[0][len(image[0])-1]
    #Bottom left corner
    elif i == new_height-1 and j == 0:
        return image[len(image)-1][0]
    # Bottom right corner
    elif i == new_height-1 and j==new_width-1:
        return image[len(image)-1][len(image[0])-1]
    #If it's not a corner
    else:
        return 0


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    """
    A function that receives an image with a single channel (a two-dimensional list) and two integers, and returns an image of size new_height x new_height so that the value of each pixel in the returned image is calculated according
    to its relative position in the source image.
    :param image: list in list[[]], source image
    :param new_height:int
    :param new_width:int
    :return:list in list[[]]
    """
    new_image = [[0 for _ in range(int(new_width))] for _ in range(int(new_height))]
    for i in range(int(new_height)):
        for j in range(int(new_width)):
            val = value_for_corners(i, j, image, new_width, new_height)
            if val == 0:
                fall_pixel = calc_fall_pixel(new_height, new_width, image, [i, j])
                new_image[i][j] = bilinear_interpolation(image, fall_pixel[0], fall_pixel[1])
            else:
                new_image[i][j] = val
    return new_image


def rotate_90(image: Image, direction: str) -> Image:
    """
    The function receives an image and direction and returns a similar image
    rotated by 90 degrees to the desired direction.
    :param image:image,list[[]]
    :param direction:str 'L' ,'R'
    :return:list [[]]
    """
    #calc the size of the source image
    columns=len(image[0])
    rows=len(image)
    #create new list for the new image
    rotate_image=[[0 for r in range(rows)] for c in range(columns)]
    for i in range(rows):
        for j in range(columns):
            if direction == 'R':
                rotate_image[j][rows-i-1]=image[i][j]
            else:rotate_image[columns-j-1][i]=image[i][j]
    return rotate_image



def mean_square_error(num_img,num_patch):
    """
    The function get current position in the image and in the patch and return the calc of
    mean square error for them.
    :return: float
    """
    return (num_img-num_patch)**2


def get_best_match(image: SingleChannelImage, patch: SingleChannelImage) -> tuple:
    """
    The function receives an image with a single color channel and a patch (a small image with a single color channel) and returns a tuple with two values - the position and distance
    of the closest patch in the image to the patch/input
    :param image: list in list[]
    :param patch: list in list[]
    :return: tuple ((x-row,y-column),float-mse)
    """
    min_mse=None
    flag=True
    closest_loc=()
    patch_rows=len(patch)
    patch_columns=len(patch[0])
    #run for each row in image
    for i in range(len(image)-patch_rows+1):
        for j in range(len(image[0])-patch_columns+1):
            sum_square=0
            for y in range(patch_rows):
                for x in range(patch_columns):
                    sum_square+=mean_square_error(image[i+y][j+x],patch[y][x])
            mse=sum_square/(len(patch)*len(patch[0]))
            if min_mse is None or mse<min_mse:
                min_mse=mse
                closest_loc=(i,j)
    return (closest_loc,min_mse)



def create_search_environment(image,patch,x,y):
    """
     Create a function that receives a large list and a smaller list and returns the original list in a truncated manner so that the
     smaller list can fit in the larger list for example.
     x and y are the range in the image.
    :return:list in list [[]]
    """
    #We will take the maximum number if there is no value to the left of the point we received
    start_row=max(x-1,0)
    start_col=max(y-1,0)
    #The calculation of the edge of the row is done by adding the length of the rows of the patch to the range
    stop_row=min(x+1+len(patch),len(image))
    #The calculation of the edge of the columns is done by adding the length of the columns of the patch to the range
    stop_col=min(y+1+len(patch[0]),len(image[0]))
    result=[]
    #Using our boundary values we will cut the list
    for i in range(start_row,stop_row):
        result.append(image[i][start_col:stop_col])
    return result



def find_patch_in_img(image: SingleChannelImage, patch: SingleChannelImage) -> dict:
    """
    A function that receives an image (has a single color channel) and a patch (two-way list) and looks
    for the approximate place of the patch in the image in several different appearances of rotations and magnifications.
    The function will return a dictionary of lists containing all the locations and distances
    :return:dict
    """
    img_row=len(image)
    img_col=len(image[0])
    patch_row=len(patch)
    patch_col=len(patch)
    dict = {0:[],90:[],180:[],270:[]}
    for i in dict:
        #When it's time to advance to the next angle and rotate the image
        if i != 0:
            patch = rotate_90(patch,"R")
        #The reduction of the original image two times three times
        img_2=resize(image,img_row//2,img_col//2)
        img_4 = resize(img_2, img_row// 4, img_col// 4)
        img_8 = resize(img_4, img_row // 8, img_col// 8)
        # The reduction of the original patch two times three times
        patch_2=resize(patch,patch_row//2,patch_col//2)
        patch_4=resize(patch_2,patch_row//4,patch_col//4)
        patch_8=resize(patch_4,patch_row//8,patch_col//8)

        #For the last reduction we will order the 8x reduced image and get mse and location
        mse8=get_best_match(img_8,patch_8)
        loc8 = mse8[0]
        x4=2*loc8[0]
        y4=2*loc8[1]

        #From now on, we will send the exact search area to the function using the x and y range
        search_environment4=create_search_environment(img_4,patch_4,x4,y4)
        mse4=get_best_match(search_environment4,patch_4)
        loc4=mse4[0]
        x2=2*loc4[0]
        y2=2*loc4[1]

        search_environment2=create_search_environment(img_2,patch_2,x2,y2)
        mse2=get_best_match(search_environment2,patch_2)
        loc2=mse2[0]
        x = 2 * loc2[0]
        y = 2 * loc2[1]
        search_environment = create_search_environment(image, patch,x,y)
        mse = get_best_match(search_environment, patch)
        dict[i]=[mse8,mse4,mse2,mse]
    return dict

if __name__ == '__main__':
    print(get_best_match([[1, 5, 1, 3], [7, 4, 6, 2], [0, 10, 2, 200], [250,9,0,240]],
                         [[7, 4, 6], [0, 10, 3], [249, 9, 1]]))
    img = load_image("img.jpg")
    patch = load_image("patch.jpg")
    img = RGB2grayscale(img)
    patch = RGB2grayscale(patch)
    print(find_patch_in_img(img, patch))