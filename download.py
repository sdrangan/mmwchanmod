"""
download.py:  Downloads files

Programs are taken from  StackOverflow answer: https://stackoverflow.com/a/39225039
"""

import os
import zipfile
import shutil
import requests
import argparse



def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    print('Downloading %s' % destination)

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)    
     

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Download files')

parser.add_argument('--model_data', dest='model_data', action='store_true',\
    help="Downloads pre-trained model")
parser.add_argument('--train_test_data', dest='train_test_data', action='store_true',\
    help="Downloads training and test data processed from the raw data")    
parser.add_argument('--raw_data', dest='raw_data', action='store_true',\
    help="Downloads raw ray tracing data")
    
args = parser.parse_args()
file_types = []
if args.model_data:
    file_types.append('model_data')
if args.train_test_data:
    file_types.append('train_test_data')
if args.raw_data:
    file_types.append('raw_data')
    
if len(file_types) == 0:
    print('No files to download.  Specify --model_data, --raw_data, or --train_test_data')
    
    
for file_type in file_types:

    
    # File IDs, destination file and extracted dir
    file_ids = {'raw_data':  '1mgvwWar7Fivwh9M7dQZDmcF0JDeZD5YE',\
                'model_data':  '1IBzVyJx_WgWDz7JInYcwQURe81Q5jSh9',
                'train_test_data': '1MbGJHHkoFoaLY0r5GNMQFjV31ZSMxWvu'}
    extract_dirs = {'raw_data':  'uav_ray_trace',\
                   'model_data':  'model_data',\
                   'train_test_data':  ''}    
    dst_fns = {'raw_data':  'uav_ray_trace.zip',\
              'model_data':  'model_data.zip',\
              'train_test_data':  'train_test.p' }
        
      
    # Download the file
    file_id = file_ids[file_type]
    dst_fn = dst_fns[file_type]
    extract_dir = extract_dirs[file_type]
    
    download_file_from_google_drive(file_id, dst_fn)
    
    # Unzip    
    if extract_dir == '':
        # Nothing to do. File is not a zip file.
        pass 
    elif os.path.exists(extract_dir):
        # Extracted directory already exists
        print('%s directory already exists.  Delete to continue' % extract_dir)
        
    else:    
        # Unzip
        print('Extracing %s' % extract_dir)
        with zipfile.ZipFile(dst_fn, 'r') as zip_ref:
            zip_ref.extractall('.')
