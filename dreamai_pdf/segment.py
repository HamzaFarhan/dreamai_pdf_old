# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_segment.ipynb.

# %% auto 0
__all__ = ['text_to_segments', 'segment_to_ners', 'ners_to_dicts', 'get_edu_dicts', 'get_job_dicts', 'get_contact_dict',
           'load_segs_model', 'load_ner_model']

# %% ../nbs/02_segment.ipynb 3
from .core import *
from .parse import *
from .imports import *


# %% ../nbs/02_segment.ipynb 4
def text_to_segments(text, labeling_model, tags=['education', 'work experience']):
    segs = defaultdict(list)
    for txt in text:
        pred = tags[labeling_model(txt, tags)[0][0]]
        segs[pred].append(txt)
    return segs

def segment_to_ners(text, tagger):
    if is_list(text):
        text = ' '.join(text)
    s = Sentence(text)
    tagger.predict(s)
    return s

def ners_to_dicts(s, search_tags=['ORG', 'DATE'], dict_keys=['COMPANY', 'DATE']):
    tags_list = []
    tags_dict = {}
    for l in s.labels:
        dp = l.data_point
        tag = dp.tag
        for s,k in zip(search_tags, dict_keys):
            if tag == s:
                if not tags_dict.get(k,None):
                    tags_dict[k] = dp.text.strip()
                else:
                    tags_list.append(tags_dict)
                    tags_dict = {k:dp.text.strip()}
                
    return tags_list

def get_edu_dicts(edu, tagger):
    edu = segment_to_ners(edu, tagger)
    edu_list = ners_to_dicts(edu, search_tags=['ORG', 'DATE'], dict_keys=['INSTITUTE', 'DATE'])
    edu_list = [d for d in edu_list if d.get('INSTITUTE', None) is not None]
    return edu_list

def get_job_dicts(job, tagger):
    job = segment_to_ners(job, tagger)
    job_dict = ners_to_dicts(job, search_tags=['ORG', 'DATE'], dict_keys=['COMPANY', 'DATE'])
    job_dict = [d for d in job_dict if d.get('COMPANY', None) is not None]
    return job_dict

def get_contact_dict(text):
    if is_list(text): text = ' '.join(text)
    mail_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    phone_regex = re.compile(r'[\d]{3}[\s-]?[\d]{3}[\s-]?[\d]{4}')
    emails = re.findall(mail_regex, text.lower())
    phones = re.findall(phone_regex, text.lower())
    return {'EMAIL':emails, 'PHONE':phones}

def load_segs_model():
    return Labels("roberta-large-mnli")

def load_ner_model():
    return Classifier.load('ner-ontonotes-large')
