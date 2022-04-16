import pandas as pd
from logger import get_logger
from dataset import IndustryDataset
from model import Model
from preprocess import Preprocess
from label_encoder import get_label_encoder
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from typing import List


class Ensemble:
    def __init__(self, test_data: pd.DataFrame,
                submission: pd.DataFrame,
                ensemble_entry: List[str]):
        self.test_data = test_data
        self.submission = submission
        self.entry = ensemble_entry
        self.categories = {}
        self.label_encoder = get_label_encoder()
        self.logger = get_logger("inference")

    def soft_ensemble(self):
        try:
            fold_preds = [self.get_fold_pred(pred) for pred in self.entry]
            sum_of_preds = F.softmax(torch.Tensor(fold_preds[0][0]), dim=1)
            for pred in fold_preds[1:]:
                sum_of_preds += F.softmax(torch.Tensor(pred[0]), dim=1)
            mean_of_preds = sum_of_preds / len(self.entry)
            outputs = mean_of_preds.argmax(axis=1)

            self._decoder(outputs)
            self.logger.info("Success ensemble and inference")
        except:
            self.logger.exception("Fail ensemble and inference")
        return self.submission

    def get_fold_pred(self, model_name):
        
        model_info = Model(model_name)
        model = model_info.get_model()
        tokenizer = model_info.get_tokenizer()

        pred_args=TrainingArguments(output_dir = "./",
                                    do_predict=True,
                                    per_device_eval_batch_size=128,)
        trainer = Trainer(model=model, args=pred_args)
        preprocesser = Preprocess()
        test = preprocesser.test_preprocess(self.test_data)

        test_dataset = IndustryDataset(test, tokenizer, is_train=False)
        preds = trainer.predict(test_dataset)
        del model
        del tokenizer
        return preds

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

