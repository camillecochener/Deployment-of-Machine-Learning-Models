from sklearn.externals import joblib

import data_management as dm
import config
import pipeline as pipe
import preprocessors as pp


def run_training(save_result: bool = True):
    
    images_df = dm.load_image_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)
    
    enc = pp.TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    
    
    pipe.pipe.fit(X_train, y_train)
    
    if save_result:
        joblib.dump(enc, config.ENCODER_PATH)
        dm.save_pipeline_keras(pipe.pipe)


if __name__ == '__main__':
    run_training(save_result=True)