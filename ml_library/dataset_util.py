import os
import random
import cv2
import numpy as np
import pickle
try:
    from .utility_functions import pre_process_image_for_vgg
    from .utility_functions import get_conversions_between_input_and_feature
    from .utility_functions import get_input_coordinates_of_anchor_points
    from .utility_functions import get_input_coordinates_of_all_anchor_boxes
    from .utility_functions import get_area_of_blobs
    from .utility_functions import get_number_of_anchor_boxes_per_anchor_point

except:
    print("You should not be executing the program from this python file...")

class dataset_generator():
    """Handles all things to do with dataset generation, saving and loading."""
    dataset_base_name = "dataset.p"

    def __init__(self,config):
        """Stores config variables and loads dataset"""
        self.backbone_model = config["BackboneClassifier"]
        self.model_input_size = config["ModelInputSize"]
        self.processed_dataset_folder = config["ProcessedDatasetFolder"]
        self.scale = config["Model"]["Scale"]
        self.aspect_ratio = config["Model"]["AspectRatio"]
        self.dataset = []

        self.load_dataset()

    def load_dataset(self):
        """Checks to see if the dataset pickle file exists. If so load it"""
        dataset_full_path = f"{self.processed_dataset_folder}{os.sep}{self.dataset_base_name}"
        if os.path.exists(dataset_full_path):
            self.dataset = pickle.load(open(dataset_full_path,"rb"))

    def has_video_already_been_processed(self,video_file_path):
        """Checks if the video has already been processed"""
        already_processed_videos_set = set([row["Meta"]["VideoPath"] for row in self.dataset])
        return video_file_path in already_processed_videos_set
             
    def save_dataset(self,mode="append new"):
        """
            Saves the dataset as a pickle file
            Checks if the folder exists, if not create the folder
            Check if the dataset exists, if not save the dataset
            If the mode is override then it will replace the file
            if the mode is append then it will check if there are new video files processed and then append
                the new video files to the existing dataset

            Returns:
                True, if new data saved
                False, if no new data is saved
        """
        # Create processed dataset folder if not existent
        if not os.path.exists(self.processed_dataset_folder):
            os.mkdir(self.processed_dataset_folder)

        # if dataset has not been saved before save
        dataset_full_path = f"{self.processed_dataset_folder}{os.sep}{self.dataset_base_name}"
        if not os.path.exists(dataset_full_path):
            pickle.dump(self.dataset, open(dataset_full_path,"wb"))
            return True

        if mode == "override": # Replace file
            pickle.dump(self.dataset, open(dataset_full_path,"wb"))
            return True
        elif mode == "append new": # Check if file contents in there and if not then append
            saved_dataset = pickle.load( open(dataset_full_path,"rb"))
            saved_dataset_processed_videos_set = set([row["Meta"]["VideoPath"] for row in saved_dataset])
            current_dataset_processed_videos_set = set([row["Meta"]["VideoPath"] for row in self.dataset])
            videos_processed_but_not_in_saved_dataset = current_dataset_processed_videos_set - saved_dataset_processed_videos_set

            if len(videos_processed_but_not_in_saved_dataset) > 0:
                additional_rows_to_save = [row for row in self.dataset if row["Meta"]["VideoPath"] in videos_processed_but_not_in_saved_dataset]
                saved_dataset.extend(additional_rows_to_save)
                pickle.dump(saved_dataset, open(dataset_full_path,"wb"))
                return True
        return False

    def get_machine_formatted_dataset(self):
        """produces a dataset in a format which ML can train on split into x and y data"""
        x_data = np.array([row["MachineFormat"]["Input"][0] for row in self.dataset])
        y_data = np.array([row["MachineFormat"]["Output"][0] for row in self.dataset])
        return x_data, y_data

    def convert_video_to_data(self,video_file_path):
        """
            Converts a video into a dataset which can be read by a human or the ML model

            Checks if the video has already been processed
            Checks that the video file can be read
            resizes the image to suit the model
            gets a mask of the object - A1
            Gets the feature map by passing the input image through the CNN backbone, in this case vgg16
            Finds conversions between the input image space and feature map space
            checks whether or not an object was found
            goes through each anchor point, creates a mask for that box and finds the iou with the box and object mask
            gets highest iou and gets the coordinate of the anchor point
            Creates the ML output matrix
            Displays images for debugging
            Saves data in a list as an array in a human and machine readable format

            Assumption 1: There is only [0,1] objects in each frame
        """
        if self.has_video_already_been_processed(video_file_path):
            return None

        cap = cv2.VideoCapture(video_file_path)
        assert cap.isOpened() # can open file
        
        index = -1
        while True:
            returned_value, frame = cap.read()
            if not returned_value:
                print("Can't receive frame (potentially stream end or end of file?). Exiting ...")
                break
            index += 1

            resized_frame = cv2.resize(frame,self.model_input_size)
            
            final_mask, final_result, object_identified = get_red_box(resized_frame)

            prediction_ready_image = pre_process_image_for_vgg(frame,self.model_input_size)
            feature_map = self.backbone_model.predict(prediction_ready_image)

            feature_to_input, input_to_feature = get_conversions_between_input_and_feature(prediction_ready_image.shape,feature_map.shape)
            coordinates_of_anchor_points = get_input_coordinates_of_anchor_points(feature_map.shape,feature_to_input)
            coordinates_of_all_anchor_boxes = get_input_coordinates_of_all_anchor_boxes(coordinates_of_anchor_points,feature_to_input,self.scale,self.aspect_ratio)

            anchor_point_overlay_display_img = final_result.copy()
            if object_identified == True:
                # TODO refactor into function
                # TODO confirm that there can only be on anchor box per object
                # TODO only assume there is one object per final mask

                iou_in_output_array = [[[dataset_generator.get_iou_from_bbox_and_mask(b_box,final_mask) for b_box in x] for x in y] for y in coordinates_of_all_anchor_boxes]
                iou_in_output_array = np.array(iou_in_output_array)

                max_location = np.where(iou_in_output_array == iou_in_output_array.max())
                height_index = max_location[0][0]
                width_index = max_location[1][0]
                bbox_channel_index = max_location[0][0]

                # Draw anchor points and boxes
                flattened_anchor_point_coords = [coord for y_coords in coordinates_of_anchor_points for coord in y_coords]
                for coord in flattened_anchor_point_coords:
                    cv2.circle(anchor_point_overlay_display_img,(coord["x"],coord["y"]),2,(0,0,255))

                # Draw anchor boxes for highlighted point
                for box in coordinates_of_all_anchor_boxes[height_index][width_index]:
                    cv2.rectangle(anchor_point_overlay_display_img,(box["x1"],box["y1"]),(box["x2"],box["y2"]),(255,255,255),1)
                # Display circle on activated anchor point
                max_anchor_point_coord = coordinates_of_anchor_points[height_index][width_index]
                cv2.circle(anchor_point_overlay_display_img,(max_anchor_point_coord["x"],max_anchor_point_coord["y"]),3,(255,255,255))
                matched_anchor_box = coordinates_of_all_anchor_boxes[height_index][width_index][bbox_channel_index]
                cv2.rectangle(anchor_point_overlay_display_img,(matched_anchor_box["x1"],matched_anchor_box["y1"]),(matched_anchor_box["x2"],matched_anchor_box["y2"]),(255,0,0),1)

                # create ground truth output for ML
                ground_truth_output = np.zeros(iou_in_output_array.shape,np.float64)
                ground_truth_output[height_index][width_index][bbox_channel_index] = 1.0
            else:
                matched_anchor_box = None

                # create ground truth output for ML
                _, height, width, _ = feature_map.shape
                anchor_box_channels = get_number_of_anchor_boxes_per_anchor_point(self.scale,self.aspect_ratio)
                ground_truth_output_shape = (height,width,anchor_box_channels)
                ground_truth_output = np.zeros(ground_truth_output_shape,dtype=np.float64)

            # Show images for debugging
            debug_image = self.gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,matched_anchor_box,feature_to_input)

            self.dataset.append({ 
                "Meta": {
                    "VideoPath" : video_file_path
                    ,"FrameIndex" : index
                }
                ,"MachineFormat" : {
                    "Input" : feature_map
                    ,"Output" : np.array([ground_truth_output])
                }
                ,"HumanFormat" : { 
                    "InputImage" : resized_frame
                    ,"ObjectMask" : final_mask
                    ,"MatchedCoord" : matched_anchor_box
                    ,"ObjectDetected" : object_identified
                    ,"AllImagesSideBySide" : debug_image
                }
                })

            print(f"[{index}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}] max iou={iou_in_output_array.max()}, coord {matched_anchor_box}")
        cv2.destroyAllWindows()

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,matched_anchor_box,feature_to_input,wait_time_ms = 10):
        """
            For debugging purposes to display all the images and returns the concatenated result
            Convert everything to colour to be compatible to be concatenated
            Note 1: The final_mask is a 2d array that must be converted into a 3d array with 3 channels
        """

        white_colour_image = np.ones(resized_frame.shape,dtype=np.uint8) * 255
        final_mask_colour = cv2.bitwise_and(white_colour_image,white_colour_image,mask=final_mask)

        debug_image = np.concatenate((resized_frame, final_mask_colour), axis=1)
        debug_image = np.concatenate((debug_image, final_result), axis=1)
        debug_image = np.concatenate((debug_image, anchor_point_overlay_display_img), axis=1)

        ground_truth_output_colour = np.zeros(resized_frame.shape,dtype=np.uint8)
        cv2.rectangle(ground_truth_output_colour,(matched_anchor_box["x1"],matched_anchor_box["y1"]),(matched_anchor_box["x2"],matched_anchor_box["y2"]),(255,255,255),-1)

        debug_image = np.concatenate((debug_image, ground_truth_output_colour), axis=1)
        
        cv2.imshow("debug_image",debug_image)
        cv2.waitKey(wait_time_ms)
        return debug_image

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def draw_anchor_point_and_boxes(img_to_draw_on,anchor_point_coord,feature_to_input,scale=None,aspect_ratio=None):
        """For debugging to see the input image with the anchor points and boxes drawn and returns bounding box"""
        # TODO: Account for scale and aspect ratio

        cv2.circle(img_to_draw_on,(anchor_point_coord["x"],anchor_point_coord["y"]),2,(0,0,255))

        bounding_box = {
            "x1" : int(round(anchor_point_coord["x"] - feature_to_input["x_offset"]))
            ,"y1" : int(round(anchor_point_coord["y"] - feature_to_input["y_offset"]))
            ,"x2" : int(round(anchor_point_coord["x"] + feature_to_input["x_offset"]))
            ,"y2" : int(round(anchor_point_coord["y"] + feature_to_input["y_offset"]))
        }
        cv2.rectangle(img_to_draw_on,(bounding_box["x1"],bounding_box["y1"]),(bounding_box["x2"],bounding_box["y2"]),(255,255,255),1)

        return bounding_box

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def get_iou_from_bbox_and_mask(b_box,final_mask):
        """Gets IOU (intersection over union) between bounding box in {x1,y1,x2,y2}  
           Accounts for bounding boxes that exceed mask dimensions, e.g. x1,y1 that are negative and x2,y2 > widht or height
           Assumption 1: Scale & aspect ratio * width & height of anchor point do not exceed the width & height of input image"""

        # Get required size of image to accommodate for bounding box's position either too far left etc...
        height, width = final_mask.shape
        x_left_offset = -b_box["x1"] if b_box["x1"] < 0 else 0
        x_right_offset = b_box["x2"] - width if b_box["x2"] > width else 0
        y_up_offset = -b_box["y1"] if b_box["y1"] < 0 else 0
        y_down_offset = b_box["y2"] - height if b_box["y2"] > height else 0

        # Make new image that fits box and mask
        new_image_height = y_up_offset + height + y_down_offset
        new_image_width = x_left_offset + width + x_right_offset
        anchor_box_mask = np.zeros((new_image_height,new_image_width), dtype=np.uint8)
        # create and slice final mask onto newly sized image
        moved_final_mask = anchor_box_mask.copy()
        moved_final_mask[y_up_offset:height + y_up_offset,x_left_offset:width + x_left_offset] = final_mask
    
        fill_constant = -1
        anchor_box_mask = cv2.rectangle(anchor_box_mask,(b_box["x1"]+x_left_offset,b_box["y1"]+y_up_offset),(b_box["x2"]+x_left_offset,b_box["y2"]+y_up_offset),255,fill_constant)

        return dataset_generator.get_iou_from_masks(anchor_box_mask, moved_final_mask)
        """ # TODO delete later, test code only to explain how extra padding is added to images 
        x_left_offset = 500
        y_up_offset = 200
        test_zeros = np.zeros((224+y_up_offset,224+x_left_offset),np.uint8)
        test_white = np.ones((224,224),np.uint8)*255
        test_zeros[y_up_offset:224+y_up_offset,x_left_offset:224+x_left_offset] = test_white
        cv2.imshow("test",test_zeros)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def create_anchor_box_mask_on_input(coord,feature_to_input,mask_shape):
        """Creates a mask where that covers the anchor box. This is used to then find the IOU of a blob"""
        assert len(mask_shape) == 2 # Should be an image [width,height] because it is a mask

        # Create empty mask
        anchor_box_mask = np.zeros(mask_shape, dtype=np.uint8)
        
        x1 = int(round(coord["x"] - feature_to_input["x_offset"]))
        y1 = int(round(coord["y"] - feature_to_input["y_offset"]))
        x2 = int(round(coord["x"] + feature_to_input["x_offset"]))
        y2 = int(round(coord["y"] + feature_to_input["y_offset"]))

        fill_constant = -1
        anchor_box_mask = cv2.rectangle(anchor_box_mask,(x1,y1),(x2,y2),255,fill_constant)

        return anchor_box_mask

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def get_iou_from_masks(single_blob_mask_1, single_blob_mask_2): # TODO find variable
        """
            Gets the Intersection Over Area, aka how much they cross over divided by the total area
            from masks (greyscale images)

            Uses bitwise or to create new mask for union blob
            Uses bitwise and to create new mask for intersection blob

            Assumption 1: The masks must be greyscale
            Assumption 2: There must only be one blob (aka object) in each mask
            Assumption 3: Both masks must be the same dimensions (aka same sized object)
            Note 1: If the union area is 0, there are no blobs hence the IOU should be 0
        """
        assert len(single_blob_mask_1.shape) == 2 # Should be a greyscale image
        assert len(get_area_of_blobs(single_blob_mask_1)) == 1 # Mask should only have one blob in it
        assert len(single_blob_mask_2.shape) == 2 # Should be a greyscale image
        assert len(get_area_of_blobs(single_blob_mask_2)) == 1 # Mask should only have one blob in it
        assert single_blob_mask_1.shape[0] == single_blob_mask_2.shape[0] and single_blob_mask_1.shape[1] == single_blob_mask_2.shape[1]

        union_mask = cv2.bitwise_or(single_blob_mask_1,single_blob_mask_2)
        if len(get_area_of_blobs(union_mask)) == 1:
            union_area = get_area_of_blobs(union_mask)[0]
        else: 
            intersection_over_union = 0.0 # Stop math error, divide by 0
            return intersection_over_union

        intersection_mask = cv2.bitwise_and(single_blob_mask_1,single_blob_mask_2)
        if len(get_area_of_blobs(intersection_mask)) == 1:
            intersection_area = get_area_of_blobs(intersection_mask)[0]
        else: 
            intersection_area = 0.0

        intersection_over_union = intersection_area / union_area
        assert intersection_over_union >= 0.0
        assert intersection_over_union <= 1.0

        return intersection_over_union

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def convert_input_image_coord_to_feature_map(coord_in_input_space,input_to_feature):
        """
            Converts an point in the input image space to the feature map space returns as a dictionary

            Assumption 1: coord_in_input_space is a dictionary of {"x",int,"y",int}
            Assumption 2: input_to_features is a dictionary
        """
        x = int(round((coord_in_input_space["x"] + input_to_feature["x_offset"])*input_to_feature["x_scale"]))
        y = int(round((coord_in_input_space["y"] + input_to_feature["y_offset"])*input_to_feature["y_scale"]))
        coord_in_feature_map = {"x":x,"y":y}
        return coord_in_feature_map

    def get_n_random_rows(self,n_random_rows):
        """Gets n random rows from the dataset"""
        return random.sample(self.dataset, n_random_rows)



def get_red_box(resized_frame,threshold_area = 400):
    """
        Uses HSV colour space to determine if a colour is actually red.
            Does this by considering the lower and upper colour space.
            Adds those two masks together
            Uses morphology to fill in the small gaps
            finds those blobs that have an area greater than threshold_area
            returns the overlayed image and the mask with only blobs greater than the threshold_area

            TODO refeactor code with new general utility functions, store away in other module for better use next timme
    """
    hsv_colour_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    #lower red
    lower_red = np.array([0,110,110])
    upper_red = np.array([10,255,255])

    #upper red
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv_colour_img, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_colour_img, lower_red2, upper_red2)

    combined_mask = cv2.bitwise_or(mask,mask2)
    
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    final_mask = mask = np.zeros(mask.shape, dtype=np.uint8)

    contours, _  = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs_greater_than_threshold = [blob for blob in contours if cv2.contourArea(blob) > threshold_area]
    for blob in blobs_greater_than_threshold:
        cv2.drawContours(final_mask, [blob], -1, (255), -1)

    final_result = cv2.bitwise_and(resized_frame,resized_frame, mask= final_mask)

    object_identified = final_mask.max() > 0

    return final_mask, final_result, object_identified




