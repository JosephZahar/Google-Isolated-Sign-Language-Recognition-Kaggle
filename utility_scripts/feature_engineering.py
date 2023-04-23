# %% [code]
import pandas as pd
import numpy as np

        
class EngineeredColumn:
    data_columns = []

    def __init__(self, col_name):
        self.col_name = col_name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.data_columns.clear()
            self.data_columns.append(self.col_name)
            return func(*args, **kwargs)
        return wrapper


class FeatureEngineer:

    def __init__(self):
        self.HANDS = [[0, 1, 2, 3, 4,],
                      [0, 5, 6, 7, 8],
                      [0, 9, 10, 11, 12],
                      [0, 13, 14, 15, 16],
                      [0, 17, 18, 19, 20]]
        self.ARMS = [[22, 16, 20, 18, 16, 14, 12, 11, 13, 15, 17, 19, 15, 21]]
        self.EYES = [[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 23],
                     [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]]
        self.LIPS = [[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]]
        self.ROWS_PER_FRAME = 543
        self.data_columns= []


    @EngineeredColumn("velocity")
    def calculate_velocity(self, dataset):
        dataset['velocity'] = np.sqrt((dataset.x.shift(-543)-dataset.x)**2
                                      + (dataset.y.shift(-543)-dataset.y)**2
                                      + (dataset.z.shift(-543)-dataset.z)**2).shift(543, fill_value=0)

    @EngineeredColumn("movement_x")
    def calculate_movement_x(self, dataset):
        dataset['movement_x'] = (dataset.x.shift(-543)-dataset.x).shift(543, fill_value=0)

    @EngineeredColumn("movement_y")
    def calculate_movement_y(self, dataset):
        dataset['movement_y'] = (dataset.y.shift(-543)-dataset.y).shift(543, fill_value=0)

    @EngineeredColumn("movement_z")
    def calculate_movement_z(self, dataset):
        dataset['movement_z'] = (dataset.z.shift(-543)-dataset.z).shift(543, fill_value=0)

    @EngineeredColumn("acceleration")
    def calculate_acceleration(self, dataset):
        dataset['acceleration'] = np.sqrt(((dataset.velocity.shift(-543)-dataset.velocity)/2)**2
                                          + ((dataset.velocity.shift(-543)-dataset.velocity)/2)**2
                                          + ((dataset.velocity.shift(-543)-dataset.velocity)/2)**2).shift(543, fill_value=0)


    def add_all_features(self, dataset):
        self.calculate_velocity(dataset)
        self.calculate_movement_x(dataset)
        self.calculate_movement_y(dataset)
        self.calculate_movement_z(dataset)
        self.calculate_acceleration(dataset)
        return dataset


    def load_engineered_data(self, dataset):
        data_columns = EngineeredColumn.data_columns
        new_data = self.add_all_features(dataset).loc[:, data_columns]
        n_frames = int(len(new_data) / self.ROWS_PER_FRAME)
        new_data = new_data.values.reshape(n_frames, self.ROWS_PER_FRAME, len(data_columns))
        return new_data.astype(np.float32)
    
    


    
#     def calculate_distance / calculate_angle (specific points?)
    
    
    
    
    