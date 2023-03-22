def data_path():

    train_path = {
                "image" : "/home/nlplab/hdd1/eunseo/AutoImageCaptioning/dataset/train2014",
                "annotation" : "/home/nlplab/hdd1/eunseo/AutoImageCaptioning/dataset/annotations/captions_train2014.json"}

    val_path = {
                "image" : "/home/nlplab/hdd1/eunseo/AutoImageCaptioning/dataset/val2014",
                "annotation" : "/home/nlplab/hdd1/eunseo/AutoImageCaptioning/dataset/annotations/captions_val2014.json"}


    # test_path ={
    #             "image" : "/home/nlplab/hdd1/eunseo/AutoImageCaptioning/dataset/test2014",
    #             "annotation" : "/home/nlplab/hdd1/eunseo/BERT/data/abstract_v002_val2015_annotations.json"}


    return train_path, val_path
