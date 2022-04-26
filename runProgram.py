import os

from tensorflow import keras

import ceramicModel


if __name__ == '__main__':
    model = keras.models.load_model("Model___Date_Time_2022_04_23__14_34_41___Loss_0.4566604495048523___Accuracy_0.800000011920929.h5")

    # Creating The Output directories if it does not exist
    output_directory = 'test2'
    os.makedirs(output_directory, exist_ok=True)

    # Setting the Window Size which will be used by the Rolling Average Process
    window_size = 25
    video_files_names_list = os.listdir("test_all")


    ### predict on one video:
    video_title = "IMG_0319"
    input_video_file_path = f'test_all/{video_title}.mp4'

    # Construting The Output YouTube Video Path
    output_video_file_path = f'{output_directory}/{video_title} -Output-WSize {window_size}.mp4'

    # Calling the predict_on_live_video method to start the Prediction.
    ceramicModel.predict_on_live_video(input_video_file_path, output_video_file_path,
                          window_size, model)


    ### predict on directory of videos:
    # for file in video_files_names_list:
    #     file = os.path.splitext(file)[0]
    #     input_video_file_path = f'test_all/{file}.mp4'
    #
    #     # Construting The Output YouTube Video Path
    #     output_video_file_path = f'{output_directory}/{file} -Output-WSize {window_size}.mp4'
    #
    #     # Calling the predict_on_live_video method to start the Prediction.
    #     ceramicModel.predict_on_live_video(input_video_file_path, output_video_file_path,
    #                           window_size, model)

    # ceramicModel.predict_on_live_camera(0,window_size, model)
