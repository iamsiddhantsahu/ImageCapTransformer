from PIL import Image
import os


def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
<<<<<<< HEAD
    image = image.resize([224, 224], Image.ANTIALIAS)
=======
    image = image.resize([229, 229], Image.ANTIALIAS)
>>>>>>> 1f589846e6dfa407077ad57147d54429b300c03f
    return image

def main():
    splits = ['train', 'val']
    for split in splits:
<<<<<<< HEAD
        folder = './image/%s2014' %split
        resized_folder = './image/%s2014_resized/' %split
=======
        folder = './data/images2014/%s2014' %split
        resized_folder = './data/images2014/%s2014_resized/' %split
>>>>>>> 1f589846e6dfa407077ad57147d54429b300c03f
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print ('Start resizing %s images.' %split)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print ('Resized images: %d/%d' %(i, num_images))


if __name__ == '__main__':
    main()
