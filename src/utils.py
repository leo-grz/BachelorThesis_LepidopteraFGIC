import matplotlib.pyplot as plt
import os


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

def check_folder_exists(folder_path, min_fileamount=10): 
    if os.path.exists(folder_path) and os.path.isdir(folder_path): 
        listdir = os.listdir(folder_path)
        if len(listdir) < min_fileamount:
            raise FileNotFoundError(f"The folder '{folder_path}' does not contain more than {min_fileamount} files.")
    else: 
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")



# def show_sample_by_sample(sample, path_to_images):

#     gbifid = sample['gbifID']
#     label = sample['scientificName']
#     original_filename = sample['identifier'].split('/')[-1]
#     image_name = f'{gbifid}_{original_filename}'

#     img_name = glob.glob(os.path.join(root_dir, f"{data_frame.iloc[idx, 0]}_*.jpg"))[0] # select the first tile matching the pattern

#     selected_fields = sample_row[['gbifids', 'column2', 'column3']]
#     image_name = 

#     image = 

#     image = image.numpy().transpose((1, 2, 0))
    
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     ax.axis('off')  # Turn off the axis

#     # Creating the text to be displayed
#     info_text = f"Labeled: {label}, {label_dec}\n"
#     info_text += f"GBIF ID: {gbifid}\n" \
#                 f"Filename: {image_name}" 

#     # Adding the text box
#     props = dict(boxstyle='square', facecolor='lightblue', alpha=0.5)
#     plt.text(1.03, 0.8, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

#     plt.show()
#     plt.pause(0.001)  