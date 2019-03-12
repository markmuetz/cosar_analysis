import sys
import os
import shutil

import simplejson
import requests
import tarfile


ARTICLE_FILE_JSON = {
    'skeleton_dir': {
        'filename': 'json_data/7832204_Climatology_of_shear_uau197_skeleton.json',
        'private_link': 'private_link=14d6d22c805c4e8a8e4b',
    },
    'um_output': {
        'filename': 'json_data/7831574_Climatology_of_shear_UM109_GA70_output_data_1988090119930601.json',
        'cd_loc': 'u-au197/share/data/history/P5Y_DP20',
        'private_link': 'private_link=6867f053762068c659a6',
    },
    'analysis_output': {
        'filename': 'json_data/7832207_Climatology_of_shear_cosarv0832_analysis_data.json',
        'cd_loc': 'u-au197',
        'private_link': 'private_link=00224e2655b11a4213ad',
    },

}

FILE_DATA = {
    'all': ['skeleton_dir', 'um_output', 'analysis_output'],
    'profiles': ['skeleton_dir', 'profiles'],
    'analysis_output': ['skeleton_dir', 'analysis_output'],
    'skeleton': ['skeleton_dir'],
}


def download_file(url, local_filename):
    print(url)
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return local_filename
         

def untargz(filename):
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()


def main():
    assert sys.argv[1] in FILE_DATA
    if sys.argv[1] == 'all':
        r = input('Warning: will download 18G of data, continue? y/[n]: ')
        if r != 'y':
            print('Exiting')
            sys.exit()

    article_file_jsons = FILE_DATA[sys.argv[1]]
    for json_file in article_file_jsons:
        filename = ARTICLE_FILE_JSON[json_file]['filename']
        with open(filename, 'r') as f:
            article_files = simplejson.load(f)

        if 'cd_loc' in ARTICLE_FILE_JSON[json_file]:
            cd_loc = ARTICLE_FILE_JSON[json_file]['cd_loc']
            orig_dir = os.getcwd()
            if not os.path.exists(cd_loc):
                os.makedirs(cd_loc)
            os.chdir(cd_loc)

        for article_file in article_files:
            print(article_file)

            private_link = ARTICLE_FILE_JSON[json_file].get('private_link', '')
            url = article_file['download_url'] 
            if private_link:
                url += '?' + private_link
		
            filename = download_file(url, article_file['name'])
            if os.path.splitext(filename)[-1] == '.tgz':
                untargz(filename)

        if 'cd_loc' in ARTICLE_FILE_JSON[json_file]:
            os.chdir(orig_dir)


if __name__ == '__main__':
    ret = main()

