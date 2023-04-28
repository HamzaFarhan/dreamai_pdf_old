# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['pdf_img_to_np', 'cid_to_char', 'col_clusters', 'ColumnCounter', 'get_n_cols', 'combine_lines', 'get_avg_gap',
           'get_max_gap', 'combine_splits', 'split_words', 'process_text', 'pdf_to_cols', 'pdf_cols_to_text',
           'pdf_to_text', 'text_to_segments', 'segment_to_ners', 'ners_to_dicts', 'get_edu_dicts', 'get_job_dicts',
           'get_contact_dict']

# %% ../nbs/00_core.ipynb 3
from .imports import *

# %% ../nbs/00_core.ipynb 4
def pdf_img_to_np(img):
    return np.array(img.annotated)

def cid_to_char(cidx):
    try:
        return chr(int(re.findall(r'\(cid\:(\d+)\)',cidx)[0]) + 29)
    except:
        return cidx

def col_clusters(data, data2=None, n_cols=3):
    n_cols = int(n_cols)
    kmeans = KMeans(n_clusters=n_cols, algorithm='elkan', random_state=42).fit(np.reshape(data,(-1,1)))
    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)).tolist()
    cols = defaultdict(list)
    if data2 is None: data2 = data
    for i,c in enumerate(kmeans.labels_):
        cols[idx.index(c)].append(data2[i])
    return cols

class ColumnCounter(KElbowVisualizer):
    def draw(self):
        pass

def get_n_cols(data, min_c=2, max_c=10, max_n_cols=3):
    model = KMeans(algorithm='elkan', random_state=42)
    visualizer = ColumnCounter(model, k=(min_c, max_c), metric='silhouette')
    visualizer.fit(np.reshape(data,(-1,1)))
    # print(np.reshape(data,(-1,1)))
    if visualizer.elbow_value_ is None:
        return max_n_cols
    return min(visualizer.elbow_value_, max_n_cols)

def combine_lines(txt):
    avg_len = np.mean([len(w.split()) for w in txt])
    txt2 = []
    for i,w in enumerate(txt):
        w = process_text(w)
        if i==0:
            txt2.append(w)
        else:
            if len(txt2[-1].split()) < avg_len:
                txt2[-1]+=' '+w
            else:
                txt2.append(w)
    return txt2

def get_avg_gap(words, key0='top', key1='bottom'):
    if key1 is None: key1 = key0
    return np.mean([w[key0]-words[i-1][key1] for i,w in enumerate(words) if i>0])

def get_max_gap(words, key0='top', key1='bottom'):
    if key1 is None: key1 = key0
    return np.max([w[key0]-words[i-1][key1] for i,w in enumerate(words) if i>0])

def combine_splits(splits):
    avg_len = np.mean([len(s) for s in splits])
    splits2 = []
    for i,s in enumerate(splits):
        if i==0:
            splits2.append(s)
        else:
            if len(splits2[-1]) < avg_len:
                splits2[-1]+=s
            else:
                splits2.append(s)
    return splits2

def split_words(words, key0='top', key1='bottom', avg_gap=None, fill_empty=False):
    if key1 is None: key1 = key0
    if avg_gap is None:
        avg_gap = np.mean([w[key0]-words[i-1][key1] for i,w in enumerate(words) if i>0])
    splits = []
    for i,w in enumerate(words):
        if i==0:
            splits.append([w])
        else:
            if w[key0]-words[i-1][key1] > avg_gap:
                if fill_empty:
                    splits.append(['*']*len(splits[-1]) + [w])
                else:
                    splits.append([w])
            else:
                splits[-1].append(w)
    return splits

def process_text(text):
    text = cid_to_char(text)
    text = re.sub(r"\uf0b7", " ", text)
    text = re.sub(r"\(cid:\d{0,3}\)", " ", text)
    text = re.sub(r'• ', " ", text)
    text = re.sub(r'● ', " ", text)
    return text

def pdf_to_cols(data_path, max_n_cols=3, cols_list=[2,1]):
    pdfs = resolve_data_path(data_path)
    cols_dict = {}
    for file in pdfs:
        if Path(file).suffix == '.pdf':
            try:
                with pdfplumber.open(file) as pdf:
                    pdf_pages = pdf.pages
                    cols_list = cols_list + [None]*(len(pdf_pages)-len(cols_list))
                    pdf_cols = []
                    for page, n_cols in zip(pdf_pages, cols_list):
                        words = page.extract_words(x_tolerance=5)
                        if len(words) == 0:
                            # raise Exception(f'\nCould not extract words from pdf: {str(file)}\nMaybe try extracting tables?')
                            print(f'\nCould not extract words from pdf: {str(file)}. Maybe try extracting tables?')
                            continue
                        word_x = [w['x0'] for w in words]
                        if n_cols is None:
                            try:
                                n_cols = get_n_cols(word_x, max_n_cols=max_n_cols)
                            except:
                                print(f'\nCould not find ideal number of columns for pdf: {str(file)}. Setting to 1.')
                                n_cols = 1
                        # if n_cols == 0:
                            # print(file)
                        cols = col_clusters(word_x, words, n_cols=n_cols)
                        cols = sort_dict({k:sorted(v, key=lambda x: x['top']) for k,v in cols.items()})
                        for k,v in cols.items():
                            paras = []
                            avg_gap = np.mean([w['top']-v[i-1]['top'] for i,w in enumerate(v) if i>0])
                            for i,w in enumerate(v):
                                txt = w['text']
                                if i==0:
                                    paras.append(txt)
                                else:
                                    if w['top']-v[i-1]['bottom'] >= avg_gap:
                                        paras.append(txt)
                                    else:
                                        paras[-1]+=' '+txt.strip()
                            paras = combine_lines(paras)                
                            cols[k] = paras
                        pdf_cols.append(cols)
                    cols = defaultdict(list)
                    for c in pdf_cols:
                        for k,v in c.items():
                            cols[k]+=v
                    cols = sort_dict(cols)
                    cols_dict[str(file)] = cols
            except:
                continue
    return cols_dict

def pdf_cols_to_text(pdf_cols):
    return flatten_list([dict_values(d) for d in dict_values(pdf_cols)])

def pdf_to_text(data_path, max_n_cols=3, cols_list=[2,1]):
    pdf_cols = pdf_to_cols(data_path, max_n_cols=max_n_cols, cols_list=cols_list)
    return pdf_cols_to_text(pdf_cols)

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
