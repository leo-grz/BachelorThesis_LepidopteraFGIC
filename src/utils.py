import matplotlib.pyplot as plt
from moth_dataset import MothDataset


def show_sample(image, label, label_dec, gbifid, image_name, predicted=None, predicted_dec=None):
    image = image.numpy().transpose((1, 2, 0))
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # Turn off the axis

    # Creating the text to be displayed
    info_text = f"Labeled: {label}, {label_dec}\n"
    if predicted and predicted_dec: info_text += f"Predicted: {predicted}, {predicted_dec}\n"
    info_text += f"GBIF ID: {gbifid}\n" \
                f"Filename: {image_name}" 

    # Adding the text box
    props = dict(boxstyle='square', facecolor='lightblue', alpha=0.5)
    plt.text(1.03, 0.8, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

    plt.show()
    plt.pause(0.001)  