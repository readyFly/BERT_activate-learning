import os
from EventExtract_trigger_cross_validation import model
from read_txt import read_txt_To_test_data

def predict_txt(txt_path):
    read_txt_To_test_data(txt_path)
    model()


