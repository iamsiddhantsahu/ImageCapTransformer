import json
import os
import pandas as pd
import pickle

def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print ("The number of captions before deletion: %d" %len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print ("The number of captions after deletion: %d" %len(caption_data))
    return caption_data

def main():

    max_length = 25 # maximum length of caption(number of word). if caption is longer than max_length, deleted.

    train_lookup_table = _process_caption_data(caption_file='./data/captions2014/captions_train2014.json', image_dir='./data/images2014/train2014_resized/', max_length=max_length)
    val_lookup_table = _process_caption_data(caption_file='./data/captions2014/captions_val2014.json', image_dir='./data/images2014/val2014_resized/', max_length=max_length)


    #print head of train_lookup_table and val_lookup_table DataFrame
    print (train_lookup_table.head(10))
    print (val_lookup_table.head(10))

    #saving DataFrame in pickle format
    train_lookup_table.to_pickle("./train_lookup_table.pkl")
    val_lookup_table.to_pickle("./val_lookup_table.pkl")

    print ("Look up table .pkl file saved to disk")

if __name__ == "__main__":
    main()
