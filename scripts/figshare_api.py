#!/usr/bin/env python
"""Upload files to articles on Figshare, based on UPLOAD_DATA

Modified version of: https://docs.figshare.com/#upload_files_example_upload_on_figshare
"""
import sys
import os
import hashlib
import json
import time
from glob import glob

import simplejson
import requests
from requests.exceptions import HTTPError


BASE_URL = 'https://api.figshare.com/v2/{endpoint}'
# TOKEN provides access to the API. Read it from file.
with open(os.path.join(os.path.expandvars('$HOME'), '.figshare_token.txt'), 'r') as f:
    TOKEN = f.read().strip()

CHUNK_SIZE = 1048576 

BASE_SUITE_DIR = '/nerc/n02/n02/mmuetz/um10.9_runs/archive/u-au197/'
BASE_DIR = '/nerc/n02/n02/mmuetz/um10.9_runs/archive/'

UPLOAD_DATA = [
    {'title': 'Climatology of shear: u-au197 skeleton',
     'files': [BASE_DIR + 'u-au197_skeleton.tgz']},
    {'title': 'Climatology of shear: cosar_v0.8.3.2 analysis data',
     'files': [BASE_SUITE_DIR + 'om_v0.11.1.0_cosar_v0.8.3.2_e889d0f4f8.tgz']},
    {'title': 'Climatology of shear: UM10.9 GA7.0 output data 19880901-19930601',
     'files': sorted(glob(BASE_SUITE_DIR + 'share/data/history/P5Y_DP20/au197a.pc19*.uvcape.nc*'))},
]


def raw_issue_request(method, url, data=None, binary=False):
    headers = {'Authorization': 'token ' + TOKEN}
    if data is not None and not binary:
        data = json.dumps(data)
    response = requests.request(method, url, headers=headers, data=data)
    try:
        response.raise_for_status()
        try:
            data = json.loads(response.content)
        except ValueError:
            data = response.content
    except HTTPError as error:
        print('Caught an HTTPError: {}'.format(error))
        print('Body:\n', response.content)
        raise

    return data


def issue_request(method, endpoint, *args, **kwargs):
    return raw_issue_request(method, BASE_URL.format(endpoint=endpoint), *args, **kwargs)


def list_articles():
    result = issue_request('GET', 'account/articles')
    print('Listing current articles:')
    if result:
        for item in result:
            print('  {url} - {title}'.format(**item))
    else:
        print('  No articles.')
    print()
    return result


def create_article(title):
    data = {
        'title': title  # You may add any other information about the article here as you wish.
    }
    result = issue_request('POST', 'account/articles', data=data)
    print('Created article:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])

    return result['id']


def list_files_of_article(article_id):
    result = issue_request('GET', 'account/articles/{}/files'.format(article_id))
    print('Listing files for article {}:'.format(article_id))
    if result:
        for item in result:
            print('  {id} - {name}'.format(**item))
    else:
        print('  No files.')

    print()
    return result


def get_file_check_data(file_name):
    with open(file_name, 'rb') as fin:
        md5 = hashlib.md5()
        size = 0
        data = fin.read(CHUNK_SIZE)
        while data:
            size += len(data)
            md5.update(data)
            data = fin.read(CHUNK_SIZE)
        return md5.hexdigest(), size


def initiate_new_upload(article_id, file_name):
    endpoint = 'account/articles/{}/files'
    endpoint = endpoint.format(article_id)

    md5, size = get_file_check_data(file_name)
    data = {'name': os.path.basename(file_name),
            'md5': md5,
            'size': size}

    result = issue_request('POST', endpoint, data=data)
    print('Initiated file upload:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])

    return result


def complete_upload(article_id, file_id):
    issue_request('POST', 'account/articles/{}/files/{}'.format(article_id, file_id))


def upload_parts(file_info, file_path):
    url = '{upload_url}'.format(**file_info)
    result = raw_issue_request('GET', url)

    print('Uploading parts:')
    with open(file_path, 'rb') as fin:
        for part in result['parts']:
            upload_part(file_info, fin, part)
    print()


def upload_part(file_info, stream, part):
    udata = file_info.copy()
    udata.update(part)
    url = '{upload_url}/{partNo}'.format(**udata)

    stream.seek(part['startOffset'])
    data = stream.read(part['endOffset'] - part['startOffset'] + 1)

    raw_issue_request('PUT', url, data=data, binary=True)
    print('  Uploaded part {partNo} from {startOffset} to {endOffset}'.format(**part))


def filename_safe(s):
    return "".join([c for c in s if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(' ', '_')


def main():
    if sys.argv[1] == 'upload':
        articles = list_articles()
        for upload in UPLOAD_DATA:
            title, files = upload['title'], upload['files']
            article_matches = [a for a in articles if a['title'] == title]
            if article_matches:
                print('Title {} already added'.format(title))
                article = article_matches[0]
                article_id = article['id']
                print(article_id)
            else:
                # We first create the article.
                article_id = create_article(title)
                articles = list_articles()
            article_files = list_files_of_article(article_id)

            # Then we upload the files.
            for file_path in files:
                if os.path.basename(file_path) in [af['name'] for af in article_files]:
                    print('File {} already uploaded'.format(file_path))
                    continue

                print('uploading file: {}'.format(file_path))
                file_info = initiate_new_upload(article_id, file_path)
                # Until here we used the figshare API; following lines use the figshare upload service API.
                upload_parts(file_info, file_path)
                # We return to the figshare API to complete the file upload process.
                complete_upload(article_id, file_info['id'])
                time.sleep(5)

            # Check file has been added.
            list_files_of_article(article_id)
    elif sys.argv[1] == 'list':
        articles = issue_request('GET', 'account/articles')
        for article in articles:
            print('{title}'.format(**article))
            article_id = article['id']
            article_files = issue_request('GET', 'account/articles/{}/files'.format(article_id))
            for article_file in sorted(article_files, key=lambda a: a['name']):
                # print(article_file)
                print('  {name} - {download_url}'.format(**article_file))
        return articles
    elif sys.argv[1] == 'gen_download_data':
        articles = issue_request('GET', 'account/articles')
        os.makedirs('json_data', exist_ok=True)
        for article in articles:
            print('{title}'.format(**article))
            if not article['title'].startswith('Climatology of shear'):
                continue
            article_id = article['id']
            article_files = issue_request('GET', 'account/articles/{}/files'.format(article_id))
            with open('json_data/{}_{}.json'.format(article_id, filename_safe(article['title'])), 'w') as f:
                f.write(simplejson.dumps(article_files))

            for article_file in sorted(article_files, key=lambda a: a['name']):
                # print(article_file)
                print('  {name} - {download_url}'.format(**article_file))


if __name__ == '__main__':
    ret = main()
