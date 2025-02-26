####
import image_editor
import ex5_helper
import math
import math

def separate_channels(image):
    """
     The function get image in rows x columns x channels x and return
     list of images that each of them represent ערוץ צבע בודד
    :param image:3D list
    :return:3D list
    """
    result=[]
    #run for each row
    for i in range(len(image[0][0])):
        channel=[]
        #run for each column(j lists that each list is a column)
        for j in range(len(image)):
            row_in_channel=[]
            #run for each pixle
            for k in range(len(image[0])):
                row_in_channel.append(image[j][k][i])
            channel.append(row_in_channel)
        result.append(channel)
    return result

def combine_channels(channels):
    """
    The function get a channels-long list of two-dimensional images consisting of individual color channels
    and combines them into a single image of dimensions rows x columns x channels
    :param image:3D list
    :return:3D list
    """
    result = []
    #run for each pixle
    for i in range(len(channels[0])):
        row = []
        # run for each column(j lists that each list is a column)
        for j in range(len(channels[0][0])):
            pixel = []
            #run for each row
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

def RGB2grayscale(colored_image):
    """
    The function receives a color image (a three-dimensional list) and returns an image
    in black and white tones (a two-dimensional list).
    :param colored_image:3D list
    :return:A two-dimensional list
    """
    #In this func we don't use three for loops because the pixel will always have three columns according
    #the A condition that maintains input integrity
    gray_ret_image=[]
    #run for each row
    for i in range(len(colored_image)):
        row=[]
        #run for each column
        for j in range(len(colored_image[0])):
            pixel=calc_RGB2grayscale(colored_image[i][j][0],colored_image[i][j][1],colored_image[i][j][2])
            #check that pixel is'nt none and we can use round func.
            if pixel!=None:
                #Take the nearest whole value
                pixel=round(pixel)
                row.append(pixel)
        gray_ret_image.append(row)
    return gray_ret_image

def blur_kernel(size):
    """
    The function accepts a positive and odd integer.
    Returns a kernel (two-dimensional list) of size x size
    where each cell has one value divided by size squared.
    :param size:int size
    :return:size x size list
    """
    result=[]
    row=[]
    for i in range(size):
        row=[]
        for j in range(size):
            row.append(1/size**2)
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
def apply_kernel(image , kernel):
    """
    A function receives an image with a single color channel (i.e. a two-dimensional list) and returns an
    image of the same size as the original image when each pixel in the new image is calculated by running the kernel on it.
    Identify the pixel [column]image[row] with the central entry in the kernel matrix, and sum the values of its neighbors
    (including the pixel itself) times their corresponding entry in the kernel.
    :param image:
    :param kernel:
    :return:
    """
    #calculate how many pixels are next to the middle number in kernel,
    #the number is the amount of the numbers from right to left and below and above.
    kernel_height=len(kernel)
    kernel_width=len(kernel[0])
    kernel_middle=kernel_height//2
    ret_image=[]
    for i in range(len(image)):
        row=[]
        #run for each column in the row and calc the new value for each place.
        for j in range(len(image[0])):
            #call the function that calc the result for multiply the kernel by the current image[i][j]
            sum_of_pixel=multiply_kernel_by_current(i, j, kernel, image)
            if sum_of_pixel<0:
                sum_of_pixel=0
            elif sum_of_pixel>255:
                sum_of_pixel=255
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

def bilinear_interpolation(image, y, x):
    """
    A function that receives an image with a single color channel (a two-dimensional list)
    and the coordinate of a pixel from the target image as they "fall" in the source image.
    :param image: source image
    :param y:row
    :param x:column
    :return:The pixel value, an integer.
    """
    pixel=[y,x]
    a,b,c,d=four_points_around(pixel)
    a_val,b_val,c_val,d_val=value_in_x_y(image,a,b,c,d)
    delta_y=y-a[0]
    delta_x=x-a[1]
    new_image=a_val*(1-delta_x)*(1-delta_y)+b_val*delta_y*(1-delta_x)+c_val*delta_x*(1-delta_y)+d_val*delta_x*delta_y
    return int(new_image)

def calc_fall_pixel(new_height,new_width,image,pixel):
    """
    The function get an image(list in list)and a pixel in destination and calc the
    value of x and y of the pixel in the source image.
    :return:pixel list in list[[]]
    """
    image_height=len(image)
    image_width=len(image[0])
    # To get the new pixel we will take the y and x values ​​and divide them by
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

def all_smaller_parts(matrix, row_size, col_size):
    """
    The function return a list with all the smaller images in size of row_size x col_size
    return:list [[[]]]
    """
    result=[]

    #run for the rows
    for i in range(len(matrix)-row_size+1):
        #run from each column
        for j in range(len(matrix[0])-col_size+1):
            row = []
            #run from one column exact time as the submatrix lenght
            for k in range(row_size):
                row.append(matrix[i+k][j:j+col_size])
            result.append(row)
    return result

def resize(image , new_height, new_width):
    """
    A function that receives an image with a single channel (a two-dimensional list) and two integers, and returns an image of size new_height x new_height so that the value of each pixel in the returned image is calculated according
    to its relative position in the source image.
    :param image: list in list[[]], source image
    :param new_height:int
    :param new_width:int
    :return:list in list[[]]
    """
    new_image=[[0 for _ in range(new_width)] for _ in range(new_height)]
    for i in range(new_height):
        for j in range(new_width):
            val=value_for_corners(i,j,image,new_width,new_height)
            if val==0:
                fall_pixel = calc_fall_pixel(new_height, new_width, image, [i, j])
                new_image[i][j] =bilinear_interpolation(image,fall_pixel[0],fall_pixel[1])
            else:
                new_image[i][j] = val
    return new_image


print(resize([[15,30,45,60,75],[90,105,120,135,150],[165,180,195,210,225]],6,7))


#print(resize([[15,30,45,60,75],[90,105,120,135,150],[165,180,195,210,225]],6,7))
print(rotate_90([[1, 2, 3], [4, 5, 6]], 'R'))
print(rotate_90([[1, 2, 3], [4, 5, 6]], 'L'))
print(rotate_90([[[1, 2, 3], [4, 5, 6]],
 [[0, 5, 9], [255, 200, 7]]], 'L'))


img = load_image("img.jpg")
    patch = load_image("patch.jpg")
    img = RGB2grayscale(img)
    patch = RGB2grayscale(patch)