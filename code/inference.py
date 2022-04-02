import numpy as np
import pandas as pd
from logger import get_logger
from dataset import IndustryDataset
from model import Model
from preprocess import Preprocess
from label_encoder import get_label_encoder
from transformers import Trainer


class Inference:
    def __init__(self, test_data: pd.DataFrame, model_name: str, submission: pd.DataFrame):
        self.test_data = test_data
        self.categories = {}
        self.model_name = model_name
        self.submission = submission
        self.label_encoder = get_label_encoder()
        self.logger = get_logger("inference")

    def inference(self):
        try:
            model_info = Model(self.model_name)
            model = model_info.get_model()
            tokenizer = model_info.get_tokenizer()

            trainer = Trainer(model=model)
            preprocesser = Preprocess()
            test = preprocesser.test_preprocess(self.test_data)

            test_dataset = IndustryDataset(test, tokenizer, is_train=False)
            preds = trainer.predict(test_dataset)
            outputs = np.argmax(preds[0], axis=-1)
            self._decoder(outputs)
            self.logger.info(f"Success inference: {self.model}")
        except:
            self.logger.exception(f"Fail inference: {self.model}")
        return self.submission

    def save_submission(self, exp_name: str):
        self.submission.to_csv(f"{exp_name}.csv", index=False)

    def _decoder(self, outputs):
        outputs = self.label_encoder.inverse_transform(outputs)
        self.test_data["label"] = outputs
        self._make_category()
        self.submission["digit_3"] = self.test_data["label"].astype(int)
        self.submission["digit_2"] = self.submission["digit_3"] // 10
        for i in range(len(self.submission)):
            self.submission.loc[i,"digit_1"] = self.categories[self.submission.loc[i,"digit_2"]] 
        self.logger.info("Success decoding")

    def _make_category(self):
        self._set_keys("A", 1,3)
        self._set_keys("B", 5,8)
        self._set_keys("C", 10,34)
        self._set_keys("D", 35,35)
        self._set_keys("E", 36,39)
        self._set_keys("F", 41,42)
        self._set_keys("G", 45,47)
        self._set_keys("H", 49,52)
        self._set_keys("I", 55,56)
        self._set_keys("J", 58,63)
        self._set_keys("K", 64,66)
        self._set_keys("L", 68,68)
        self._set_keys("M", 70,73)
        self._set_keys("N", 74,76)
        self._set_keys("O", 84,84)
        self._set_keys("P", 85,85)
        self._set_keys("Q", 86,87)
        self._set_keys("R", 90,91)
        self._set_keys("S", 94,96)
        self._set_keys("T", 97,98)
        self._set_keys("U", 99,99)

    def _set_keys(self, alpha, code_start, code_end):
        for i in range(code_start, code_end+1):
            self.categories[i] = alpha