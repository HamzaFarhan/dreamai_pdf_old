import pdfplumber
from flair.nn import Classifier
from flair.data import Sentence
from sklearn.cluster import KMeans
from collections import defaultdict
from txtai.embeddings import Embeddings
from yellowbrick.cluster import KElbowVisualizer
from txtai.pipeline import Extractor, Entity, Labels
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from dreamai.core import *
from dreamai.vision import *
from dreamai.imports import *
from dreamai_dl.imports import *