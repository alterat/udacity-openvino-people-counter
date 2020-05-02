OUTPUT_SHAPE = (100,7)

def read_labels(fname):
    # Load list in memory
    with open(fname, 'r') as f:
        labels = f.readlines()

    # remove \n at the end
    labels = [l.strip() for l in labels]

    # add dummy value at the beginning to match model output
    labels = ['_dummy'] + labels
    return labels

def process_output(preds):
    '''
    Process output blob from ssd_inception_v2 model

    Format of output:
    img_id, label_id, x, y, width, height, score
    '''

    # Fill in if you want more objects
    return

def extract_people(preds):
    '''
    Extract people predictions from output blob.
    '''
    if preds.shape != OUTPUT_SHAPE:
        preds = preds.squeeze()

    return preds[preds[:,1]==1]


if __name__ == "__main__":
    labels = read_labels('./coco-labels-paper.txt')
    print(labels)