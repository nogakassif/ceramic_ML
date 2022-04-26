import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling2D, Dense
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from playsound import playsound

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import simpleaudio as sa

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

image_height, image_width = 32, 32
model_output_size = 2
classes_list = ["fall", "stable"]

def frames_extraction(video_path):
    # Empty List declared to store video frames
    frames_list = []

    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file
        success, frame = video_reader.read()

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Closing the VideoCapture object and releasing all resources.
    video_reader.release()

    # returning the frames list
    return frames_list


def create_dataset():
    # Declaring Empty Lists to store the features and labels values.
    temp_features = []
    features = []
    labels = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')

        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(dataset_directory, class_name))

        # Iterating through all the files present in the files list
        for file_name in files_list:
            # Construct the complete video path
            video_file_path = os.path.join(dataset_directory, class_name,
                                           file_name)

            # Calling the frame_extraction method for every video file path
            frames = frames_extraction(video_file_path)

            # Appending the frames to a temporary list.
            temp_features.extend(frames)

        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, max_images_per_class))

        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)

        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels

# a function that will construct the model
def create_model():
    # Sequential model for model construction
    model = Sequential()

    # Defining The Model Architecture
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     input_shape=(image_height, image_width, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation='softmax'))

    # Printing the models summary
    model.summary()

    return model

def plot_metric(metric_name_1, metric_name_2, plot_name, model =create_model()):
    # Get Metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    plt.clf()
    # Constructing a range object which will be used as time
    epochs = range(len(metric_value_1))

    # Plotting the Graph
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    # Adding title to the plot
    plt.title(str(plot_name))

    # Adding legend to the plot
    plt.legend()

    plt.savefig(f'plot_{metric_name_1}_{metric_name_2}.png')


# add prediction on given video files
def predict_on_live_video(video_file_path, output_file_path, window_size, model =create_model()):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path,
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   24, (original_video_width,
                                        original_video_height))

    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # image_height, image_width = 32, 32

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = \
            model.predict(np.expand_dims(normalized_frame, axis=0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(
            predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(
                axis=0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(
                predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()


class AudioPlayer:
    def __init__(self):
        self.play_obj = None

    def play(self, audiofile):
        filename = audiofile
        wave_obj = sa.WaveObject.from_wave_file(filename)
        self.play_obj = wave_obj.play()


    def is_done(self):
        if self.play_obj:
            return not self.play_obj.is_playing()
        return True


def predict_on_live_camera(camera_path, window_size, model = create_model()):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(camera_path)

    player = AudioPlayer()
    while True:
        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = \
            model.predict(np.expand_dims(normalized_frame, axis=0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(
            predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(
                axis=0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(
                predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

            # Overlaying Class Name Text On top of the Frame
            cv2.putText(frame, predicted_class_name, (80, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
            if predicted_class_name == "fall":
                print("fall")
                if player.is_done():
                    player.play("tibetan-singing-bowl.wav")
            else:
                print("stable")


        cv2.imshow('Predicted Frames', frame)

        cv2.waitKey(1)

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()



if __name__ == '__main__':
    max_images_per_class = 25

    dataset_directory = "DATA"
    classes_list = ["fall", "stable"]

    model_output_size = len(classes_list)

    features, labels = create_dataset()
    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, one_hot_encoded_labels, test_size=0.2, shuffle=True,
        random_state=seed_constant)

    # Calling the create_model method
    model = create_model()

    print("Model Created Successfully!")


    # Adding the Early Stopping Callback to the model which will continuously monitor the validation loss metric for every epoch.
    # If the models validation loss does not decrease after 15 consecutive epochs, the training will be stopped and the weight which reported the lowest validation loss will be retored in the model.
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15,
                                            mode='min', restore_best_weights=True)

    # Adding loss, optimizer and metrics values to the model.
    model.compile(loss='categorical_crossentropy', optimizer='Adam',
                  metrics=["accuracy"])

    # Start Training
    model_training_history = model.fit(x=features_train, y=labels_train, epochs=50,
                                       batch_size=4, shuffle=True,
                                       validation_split=0.2,
                                       callbacks=[early_stopping_callback])

    # Evaluate your trained model on the feature's and label's test sets
    model_evaluation_history = model.evaluate(features_test, labels_test)

    # Creating a useful name for our model, incase you're saving multiple models (OPTIONAL)
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt,
                                                    date_time_format)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

    # Saving your Model
    model.save(model_name)
    print("model saved")

    plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')

    plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
    print("hi2")


    # Creating The Output directories if it does not exist
    output_directory = 'test1'
    os.makedirs(output_directory, exist_ok=True)

    video_title = "IMG_0318"
    # Getting the YouTube Video's path you just downloaded
    input_video_file_path = f'{output_directory}/{video_title}.mp4'

    # Setting the Window Size which will be used by the Rolling Average Process
    window_size =1

    # Construting The Output YouTube Video Path
    output_video_file_path = f'{output_directory}/{video_title} -Output-WSize {window_size}.mp4'

    # Calling the predict_on_live_video method to start the Prediction.
    predict_on_live_video(input_video_file_path, output_video_file_path,
                          window_size)
    # Calling the predict_on_live_camera method to start the Prediction in real time.
    predict_on_live_camera(0, window_size)

