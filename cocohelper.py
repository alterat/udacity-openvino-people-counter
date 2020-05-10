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
    img_id, label_id, score, x1, y1, x2, y2

    Note: this is different from the format specified in the COCO main page.
    '''

    # Fill in if you want more objects
    return

def extract_people(preds):
    '''
    Extract people predictions from output blob.

    People have label_id==1
    '''
    if preds.shape != OUTPUT_SHAPE:
        preds = preds.squeeze()

    return preds[preds[:,1]==1]


if __name__ == "__main__":
    labels = read_labels('./coco-labels-paper.txt')
    print(labels)