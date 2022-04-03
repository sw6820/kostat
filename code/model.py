from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from logger import get_logger
from tokenization_kobert import KoBertTokenizer # 같은 위치에 tokenization_kobert.py 

class Model:
    """
    Get pretrained_model from HugginFace or ./model_loc
    """
    def __init__(self,name, model_type="pre_trained"):
        self.model_name = name
        self.model_type = model_type
        self.logger = get_logger("model")

    def get_model(self):
        try:
            if self.model_type == "custom":
                # If we need custom models, this part is to be added
                model = AutoModel.from_pretrained(self.model_name)
            else:
                config = AutoConfig.from_pretrained(self.model_name)
                config.num_labels = 232
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            self.logger.info(f"Success Loading Model: {self.model_name}")
        except:
            self.logger.exception(f"Fail Loading Model: {self.model_name}")
        return model

    def get_tokenizer(self):
        try:
            if self.model_name == "monologg/kobert" or self.model_name == "monologg/distilkobert":
                tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert") 
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info(f"Success Loading tokenizer: {self.model_name}")
        except:
            self.logger.exception(f"Fail Loading tokenizer: {self.model_name}")
        return tokenizer